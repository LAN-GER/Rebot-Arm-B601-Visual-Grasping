"""Shared snapshot planning and live sessions through the original B601 actions."""
from dataclasses import asdict, replace
import json
import time
from pathlib import Path

import numpy as np

from rebot_grasp.core.types import Capture
from rebot_grasp.pipeline.capture import capture_stationary
from rebot_grasp.visualization.overlay import draw_candidates, draw_detection_boxes

from rebot_grasp.cameras.offline import save_recording
from rebot_grasp.geometry.tool_frames import canonicalize_gripper_plan
from rebot_grasp.perception.target_selection import select_target
from rebot_grasp.pipeline.planning import plan_capture
from rebot_grasp.visualization.export import save_result


def infer_snapshot(capture, detector, backend, config, output, robot=None):
    targets = [] if config.full_scene else detector.detect(capture.frame)
    result = plan_capture(capture, targets, backend, config)
    plans = [canonicalize_gripper_plan(plan) for plan in result.plans]
    result = replace(result, plans=plans)
    selected = None
    if robot is not None:
        selected = next((plan for plan in plans
                         if robot.reachable(plan.T_base_pregrasp)
                         and robot.reachable(plan.T_base_grasp)), None)
        if plans and selected is None:
            result = replace(result, status="no_reachable_candidate")
    target = None if config.full_scene else select_target(targets, config.target_class)
    output = Path(output)
    save_recording(output / "frame.npz", capture, target)
    save_result(output, capture, result, target_class=config.target_class)
    return result, selected


def execute_snapshot(capture, plan, robot, output):
    """Record an attempt before dispatch so failures cannot look like a preview."""
    if capture.synthetic or plan.candidate.frame_id != capture.frame.frame_id:
        raise ValueError("Execution requires a matching live capture")
    report = {"frame_id": capture.frame.frame_id, "execution_attempted": True,
              "call_completed": False, "object_held": False, "error": None,
              "field_success": None, "result_source": "controller_return_without_field_confirmation",
              "T_base_grasp": plan.T_base_grasp.tolist(),
              "T_base_pregrasp": plan.T_base_pregrasp.tolist()}
    path = Path(output) / "execution.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        held = robot.execute(plan)
        report.update(call_completed=True, object_held=bool(held))
        return held
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run_live(camera, backend, detector, config, output, robot=None, hand_eye=None, *,
             snapshot_handler=None, snapshot_renderer=None, window_name="AnyGrasp",
             capture_handler=capture_stationary, diagnostics=None, live_preview=None,
             freeze_after_snapshot=True, diagnose_gripper=False):
    import cv2

    output = Path(output)
    if diagnose_gripper and robot is None:
        raise ValueError("Gripper diagnostic requires a live robot")
    output.mkdir(parents=True, exist_ok=False)
    save_session_settings(output, config, robot is not None and not diagnose_gripper, hand_eye,
                          diagnose_gripper=diagnose_gripper)
    window = window_name
    snapshot_handler = snapshot_handler or infer_snapshot
    snapshot_index = 0
    frozen = None
    status = "G=infer and grasp  R=resume  Q=release/home/exit" if robot else "G=infer  R=resume  Q=exit"
    if diagnose_gripper:
        status = "G=infer and test opening only  Q=release/home/exit"
    switch_keys = {}
    if config.backend in ("anygrasp", "anygrasp_filtered") and not config.full_scene and live_preview is not None:
        switch_keys = {ord(str(i)): name for i, name in enumerate(config.yolo.get("classes", [])[:9], 1)}
    try:
        # Authenticate and load the model before starting robot control.
        backend.start()
        if detector is not None:
            detector.start()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        if diagnostics is not None:
            diagnostics.start()
        camera.open()
        for _ in range(10):
            camera.capture()
        if robot is not None:
            robot.start()
        while True:
            if diagnostics is not None:
                diagnostics.poll()
            frame = camera.capture()
            if live_preview is not None and frozen is None:
                ready = live_preview.tick(frame)
                if ready is not None and diagnostics is not None:
                    diagnostics.update(ready.capture, ready.backend_view, ready.result)
            latest = live_preview.latest if live_preview is not None else None
            if frozen is not None:
                display = frozen.copy()
                caption = 'SNAPSHOT | ' + status
            elif latest is not None:
                display = latest.image.copy()
                age = max(0., time.monotonic() - latest.capture.frame.captured_at)
                caption = f'LIVE {1/max(latest.duration_s, 1e-6):.1f} infer/s  frame age {age:.2f}s | ' + status
            else:
                display = frame.color_bgr.copy()
                caption = ('LIVE detecting... | ' if live_preview is not None else '') + status
            cv2.putText(display, caption, (8, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            if switch_keys:
                choices = "  ".join(f"{chr(key)}={name}" for key, name in switch_keys.items())
                cv2.putText(display, f"Target: {config.target_class or 'auto'}  |  {choices}",
                            (8, display.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xff
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in switch_keys:
                live_preview.pause()
                config = replace(config, target_class=switch_keys[key])
                live_preview.config = config
                frozen = None
                print(f"Grasp target: {config.target_class}", flush=True)
                continue
            if key in (ord("r"), ord("R")):
                frozen = None
            if key in (ord("g"), ord("G")):
                if live_preview is not None:
                    live_preview.pause()
                # Every G acquires a new frame, including while the display is frozen.
                capture = (capture_handler(camera, robot, hand_eye) if robot
                           else Capture(camera.capture()))
                if capture.T_base_camera is not None:
                    compensation = config.calibration.get("hand_eye_compensation_m", {})
                    offset = np.eye(4)
                    offset[:3, 3] = [compensation.get(axis, 0.) for axis in ("x", "y", "z")]
                    capture = Capture(capture.frame, offset @ capture.T_base_camera)
                snapshot_index += 1
                folder = output / f"snapshot-{snapshot_index:03d}"
                result, plan = snapshot_handler(capture, detector, backend, config, folder, robot)
                if diagnostics is not None:
                    diagnostics.update(capture, backend, result)
                if snapshot_renderer:
                    frozen = snapshot_renderer(capture, backend, result)
                else:
                    frozen = (draw_candidates(capture.frame, [plan.candidate], label="SELECTED FOR EXECUTION")
                              if plan is not None else draw_candidates(capture.frame, result.candidates))
                    if not config.full_scene:
                        target = select_target(result.detections, config.target_class)
                        draw_detection_boxes(frozen, result.detections, target)
                print(f"{result.status}: {len(result.candidates)} candidates, {len(result.plans)} plans",
                      flush=True)
                if result.pointcloud_report is not None and not result.pointcloud_report['quality']['passed']:
                    reasons = result.pointcloud_report['quality']['reasons']
                    print('Point cloud rejected: ' + ', '.join(reasons), flush=True)
                if plan is not None and diagnose_gripper:
                    if capture.synthetic or plan.candidate.frame_id != capture.frame.frame_id:
                        raise ValueError("Gripper diagnostic requires a matching live capture")
                    print("[Diagnostic] Opening at ready pose", flush=True)
                    robot.gripper.open_gripper()
                    robot.gripper.confirm_open()
                    print("[Diagnostic] Opening confirmed", flush=True)
                    break
                if plan is not None:
                    cv2.imshow(window, frozen)
                    cv2.waitKey(1)
                    held = execute_snapshot(capture, plan, robot, folder)
                    print(f"B601 grasp returned object_held={held}", flush=True)
                if not freeze_after_snapshot:
                    frozen = None
    finally:
        try:
            if robot is not None:
                robot.close()
        finally:
            try:
                camera.close()
            finally:
                try:
                    if live_preview is not None:
                        live_preview.close()
                finally:
                    try:
                        backend.close()
                    finally:
                        try:
                            if diagnostics is not None:
                                diagnostics.close()
                        finally:
                            cv2.destroyAllWindows()


def save_session_settings(output, config, execute, hand_eye=None, *, diagnose_gripper=False):
    """Store resolved runtime settings, including overrides and calibration used."""
    payload = dict(config=asdict(config), execute=execute,
                   T_tcp_camera=None if hand_eye is None else hand_eye.tolist())
    if diagnose_gripper:
        payload["mode"] = "gripper_diagnostic_at_ready"
    (Path(output) / 'session.json').write_text(json.dumps(
        payload, indent=2, allow_nan=False,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)) + '\n')
