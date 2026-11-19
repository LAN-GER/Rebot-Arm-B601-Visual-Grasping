"""Run the existing checked calibration collector and keep its result in the run log."""

from functools import partial
from pathlib import Path
import runpy
import sys


WORKSPACE = Path(__file__).resolve().parents[2]


def check_display() -> None:
    # Initialize NumPy from LJ before loading the workspace's OpenCV wheel.
    from importlib import import_module
    import_module("numpy")

    wheel_dir = str(WORKSPACE / ".venv" / "lib" / "python3.10" / "site-packages")
    sys.path.insert(0, wheel_dir)
    try:
        import cv2
    finally:
        sys.path.remove(wheel_dir)

    window = "B601 calibration display check"
    cv2.namedWindow(window)
    cv2.destroyWindow(window)
    print("OpenCV status window available.", flush=True)


def main() -> None:
    if sys.argv[1:] == ["--check-display"]:
        check_display()
        return
    if "--execute" in sys.argv:
        if not sys.stdin.isatty():
            raise RuntimeError("Use a terminal so the calibration collector can accept q/c")
        # The collector opens its window after enabling the arm.
        check_display()

    sys.path.insert(0, str(WORKSPACE / "src"))
    from rebot_grasp.calibration.stationary_capture import capture_stationary

    root = WORKSPACE.parent / "rebot-arm-b601-robstride"
    collector_root = root / "work" / "wiki_grasp"
    sdk = root / "third_party" / "reBotArm_control_py"
    sys.path[:0] = [str(collector_root), str(sdk)]
    from reBotArm_control_py import controllers
    from rebot_grasp.calibration.trajectory import make_trajectory_planner

    if not Path(controllers.__file__).resolve().is_relative_to(sdk):
        raise RuntimeError("Unexpected reBotArm SDK import")
    collector = runpy.run_path(str(collector_root / "scripts" / "collect_handeye_eih.py"))
    duration = collector["AUTO_MOVE_DURATION_S"]
    from calibration.hand_eye import HandEyeCalibrator

    save = HandEyeCalibrator.save
    active_calibrations = (collector_root / "config" / "calibration").resolve()

    def save_run_candidate(result, path) -> None:
        output = Path(path).resolve()
        if output.is_relative_to(active_calibrations) and output.name == "hand_eye.npz":
            # The wrapper already saves a separate candidate in the run directory.
            print("Active calibration left unchanged; saving run candidate.", flush=True)
            return
        save(result, output)

    HandEyeCalibrator.save = staticmethod(save_run_candidate)
    wrapper = runpy.run_path(str(root / "scripts" / "wiki_calibration.py"))
    # The names below belong to the separately versioned calibration controller.
    from src.calibration.wiki_motion import MotionPlan

    planner = make_trajectory_planner(
        wrapper["WikiMotionPlanner"], root / "runtime/wiki_sdk/robot_with_payload.urdf",
        MotionPlan, wrapper["SafetyError"],
    )
    collect = partial(capture_stationary, motion_fault=lambda arm: getattr(arm, "_wiki_motion_fault", None))
    print(f"Calibration motion: {duration:g} s/pose, SDK {sdk}", flush=True)
    wrapper["main"](capture_stationary=collect, planner_type=planner, duration=duration)


if __name__ == "__main__":
    main()
