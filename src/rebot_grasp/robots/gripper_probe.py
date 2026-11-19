"""One unloaded opening trial on the B601-RS gripper using only motor 7."""

import argparse
import math
import time
from contextlib import ExitStack

import yaml

from rebot_grasp.config import load_profile
from rebot_grasp.paths import PROJECT_ROOT
from rebot_grasp.robots.gripper_driver import selected_hardware_yaml


MOTOR_ID = 7
# Match the B601 RS full-open soft stop, MIT position gains and arrival tolerance.
OPEN_RAD = 4.90
TOLERANCE_RAD = 0.12
KP = 5.0
KD = 1.0


def hardware_config():
    config = load_profile(PROJECT_ROOT / "config/anygrasp.yaml")
    repo = PROJECT_ROOT / config["robot"]["repo_root"]
    hardware = yaml.safe_load(selected_hardware_yaml(str(repo)).read_text())
    joints = [j for j in hardware["joints"] if j["motor_id"] == MOTOR_ID]
    if (hardware["groups"]["gripper"]["joints"] != ["gripper"] or len(joints) != 1
            or joints[0]["name"] != "gripper" or joints[0]["vendor"] != "robstride"
            or joints[0]["model"] != "rs-00" or "gripper" in hardware["groups"]["arm"]["joints"]):
        raise ValueError("Expected dedicated RobStride RS-00 gripper at ID 7")
    settings = config["robot"]["gripper"]["rs"]
    if settings["counterclockwise"] is not False or settings["angle_open"] != 5.0:
        raise ValueError("RS opening settings differ from the verified +4.90 rad soft stop")
    return hardware["channel"], joints[0]["feedback_id"], joints[0]["model"]


def sample(controller, motor):
    # mechPos is a synchronous parameter read. A cached state alone cannot prove arrival.
    actual = motor.robstride_get_param_f32(0x7019, 150)
    controller.poll_feedback_once()
    fault, warning = motor.robstride_get_fault_report()
    snapshot = motor.get_state_snapshot()
    age = snapshot.feedback_age_s if snapshot else None
    # This MotorBridge release exposes fault_code in get_state(), but omits
    # it when constructing the Python snapshot. Read it from the state API.
    state = motor.get_state() if snapshot and 0 <= age < 0.25 else None
    code = state.fault_code if state else None
    if fault or warning or code:
        raise RuntimeError(f"Gripper fault/warning: state={code}, report={fault}, warning={warning}")
    if not math.isfinite(actual) or not -TOLERANCE_RAD <= actual <= OPEN_RAD + TOLERANCE_RAD:
        raise RuntimeError(f"Encoder outside expected opening range: {actual} rad")
    return actual


def open_once(controller, motor, host_id, mit_mode):
    if motor.robstride_ping_host_id(host_id, 250)[0] != MOTOR_ID:
        raise RuntimeError("Motor ID 7 did not respond as expected")

    initial = sample(controller, motor)
    if abs(initial - OPEN_RAD) < TOLERANCE_RAD:
        return "already_open"
    # Scope cleanup to this handle even if a mode/enable transaction fails partway.
    try:
        motor.ensure_mode(mit_mode, 1000)
        motor.enable()
        start = time.monotonic()
        actual = sample(controller, motor)
        initial = actual
        while time.monotonic() - start < 6.0:
            # A ramp with bounded position error avoids pushing a stalled jaw
            # with the full opening error. Only advance as the encoder follows.
            command = min(OPEN_RAD, initial + 1.5 * (time.monotonic() - start),
                          actual + 0.20)
            motor.send_mit(command, 0.0, KP, KD, 0.0)
            time.sleep(0.03)
            actual = sample(controller, motor)
            if actual < initial - TOLERANCE_RAD:
                raise RuntimeError("Encoder moved in closing direction during open command")
            if command == OPEN_RAD and abs(actual - OPEN_RAD) < TOLERANCE_RAD:
                return "encoder_open"
        raise TimeoutError(f"Gripper did not reach {OPEN_RAD:.2f} rad; last reading {actual:.3f} rad")
    finally:
        motor.disable()


def controller_open_once(controller, motor, host_id):
    """Run the unmodified B601 opening state machine with only motor 7 attached."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from rebot_grasp.robots.gripper_driver import GraspDriver, ensure_rebot_sdk_in_syspath

    config = load_profile(PROJECT_ROOT / "config/anygrasp.yaml")["robot"]
    repo = ensure_rebot_sdk_in_syspath(config["repo_root"])
    from reBotArm_control_py.actuator import RebotArm

    # Construction only loads SDK configuration. Calling arm.connect() would
    # attach six other motors. Register just our already-open gripper handle.
    arm = RebotArm()
    group = arm.groups["gripper"]
    group._mm["gripper"] = motor
    group._cm["robstride"] = controller
    driver = GraspDriver(arm, SimpleNamespace(_running=True), config["gripper"], str(repo))
    driver._gripper_motor = motor
    if motor.robstride_ping_host_id(host_id, 250)[0] != MOTOR_ID:
        raise RuntimeError("Motor ID 7 did not respond as expected")
    sample(controller, motor)
    send = motor.send_mit
    send_errors = []

    def observed_send(pos, vel, kp, kd, tau):
        try:
            send(pos, vel, kp, kd, tau)
        except Exception as exc:
            send_errors.append(str(exc))
            raise

    with ThreadPoolExecutor(max_workers=1) as pool, ExitStack() as cleanup:
        motor.send_mit = observed_send
        cleanup.callback(setattr, motor, "send_mit", send)
        cleanup.callback(motor.disable)
        if not group.mode_mit():
            raise RuntimeError("B601 gripper group could not select MIT mode")
        motor.enable()
        driver._prime_gripper_state()
        sample(controller, motor)
        opening = pool.submit(driver.open_gripper)
        next_sample = time.monotonic()
        while not opening.done():
            tick = time.monotonic()
            controller.poll_feedback_once()
            driver.gripper_tick()
            if send_errors:
                raise RuntimeError(send_errors[0])
            if tick >= next_sample:
                sample(controller, motor)
                next_sample = tick + 0.05
            time.sleep(max(0.0, 1.0 / arm._rate - (time.monotonic() - tick)))
        opening.result()
        actual = sample(controller, motor)
        if not driver._position_done() or abs(actual - OPEN_RAD) >= driver._arrive_tol:
            raise RuntimeError(f"B601 returned without confirmed opening: mechPos={actual:.3f} rad")
        return "encoder_open"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--controller-open", action="store_true", help="Reproduce the B601 3-second opening call on motor 7")
    mode.add_argument("--open", action="store_true",
                        help="Open motor 7 once and disable it at the end")
    args = parser.parse_args()
    from motorbridge import Controller, Mode

    channel, host_id, model = hardware_config()
    try:
        with ExitStack() as stack:
            controller = Controller(channel)
            stack.callback(controller.close)
            stack.callback(controller.close_bus)
            motor = controller.add_robstride_motor(MOTOR_ID, host_id, model)
            stack.callback(motor.close)
            if args.controller_open:
                outcome = controller_open_once(controller, motor, host_id)
            else:
                outcome = open_once(controller, motor, host_id, Mode.MIT)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", flush=True)
        return 1
    print(outcome, flush=True)
    return 0
