from rebot_grasp.pipeline import commands
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rebot_grasp.config import load_config
from rebot_grasp.core.types import Capture
from rebot_grasp.geometry.transforms import transform, pose6d
from rebot_grasp.grasping.backends.anygrasp import AnyGraspBackend
from rebot_grasp.pipeline.anygrasp import infer_snapshot, execute_snapshot
from rebot_grasp.robots.b601 import B601Session


class Records:
    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def predict(self, points, options):
        self.calls.append((points, options))
        return self.entries


@pytest.fixture
def live_config():
    # These robot integration cases exercise full-scene planning explicitly.
    return replace(load_config(Path(__file__).resolve().parents[1] / "config/anygrasp.yaml"),
                   full_scene=True, target_class=None)


def test_sdk_to_robot_poses_and_depth_once(scene, live_config, tmp_path):
    capture, target = scene
    capture = Capture(capture.frame, transform(translation=[0.1, 0.0, -0.2]))
    # Tool X points downward, Y is jaw opening. SDK insertion depth is 2 cm.
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    records = Records([dict(rotation=rotation, translation=[.2, .0, .5],
                            depth=.02, width=.04, score=.9)])
    backend = AnyGraspBackend(records)
    config = replace(live_config, full_scene=False, target_class=target.label)
    detector = SimpleNamespace(detect=lambda frame: [target])
    robot = SimpleNamespace(reachable=lambda pose: True)
    result, plan = infer_snapshot(capture, detector, backend, config, tmp_path, robot)
    assert len(result.plans) == 1
    assert np.allclose(plan.T_base_grasp[:3, 3], [.3, 0, .28])
    assert np.allclose(plan.T_base_pregrasp[:3, 3], [.3, 0, .36])
    assert records.calls[0][1]["region_steering"].any()
    assert (tmp_path / "frame.npz").exists()

    calls = []
    session = B601Session.__new__(B601Session)
    session.controller, session.gripper = object(), object()
    session.verify_open = True
    session.config = config.robot
    session.actions = SimpleNamespace(execute_grasp=lambda *args, **kw: calls.append((args, kw)) or True)
    assert execute_snapshot(capture, plan, session, tmp_path)
    args, kwargs = calls[0]
    assert args[:2] == (session.controller, session.gripper)
    assert args[2] == pose6d(plan.T_base_grasp)
    assert args[3] == pose6d(plan.T_base_pregrasp)
    assert kwargs == {"dry_run": False, "verify_open": True}
    report = json.loads((tmp_path / "execution.json").read_text())
    assert report["execution_attempted"] and report["object_held"]


def test_candidate_selection_filters_width_then_checks_ik(scene, live_config, tmp_path):
    capture = Capture(scene[0].frame, np.eye(4))
    entries = [dict(rotation=np.eye(3), translation=[x, 0, .3],
                    depth=0, width=width, score=score)
               for x, width, score in [(.2, .10, 1.), (.3, .04, .9), (.4, .04, .8)]]
    backend = AnyGraspBackend(Records(entries))
    robot = SimpleNamespace(reachable=lambda pose: pose[0, 3] > .31)
    result, plan = infer_snapshot(capture, None, backend, live_config, tmp_path, robot)
    assert len(result.candidates) == 2
    assert plan.candidate.score == .8


def test_preview_does_not_need_tcp_or_robot(scene, live_config, tmp_path):
    backend = AnyGraspBackend(Records([dict(rotation=np.eye(3), translation=[0, 0, .4],
                                           depth=.02, width=.04, score=.9)]))
    result, selected = infer_snapshot(Capture(scene[0].frame), None, backend, live_config, tmp_path)
    assert selected is None
    assert result.plans == []
    assert len(result.candidates) == 1
    assert not (tmp_path / "execution.json").exists()


def test_inference_error_never_calls_robot(scene, live_config, tmp_path):
    backend = SimpleNamespace(predict=lambda *args: (_ for _ in ()).throw(RuntimeError("SDK failed")))
    robot = SimpleNamespace(reachable=lambda pose: pytest.fail("IK called after inference error"))
    with pytest.raises(RuntimeError, match="SDK failed"):
        infer_snapshot(scene[0], None, backend, live_config, tmp_path, robot)


def test_live_authentication_failure_closes_resources_before_motion(monkeypatch, live_config, tmp_path):
    from rebot_grasp.pipeline.anygrasp import run_live
    import cv2
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    events = []
    def fail():
        raise RuntimeError("license invalid")
    backend = SimpleNamespace(start=fail, close=lambda: events.append("backend closed"))
    camera = SimpleNamespace(open=lambda: pytest.fail("camera opened"),
                             close=lambda: events.append("camera closed"))
    robot = SimpleNamespace(start=lambda: pytest.fail("robot enabled"),
                            close=lambda: events.append("robot closed"))
    with pytest.raises(RuntimeError, match="license invalid"):
        run_live(camera, backend, None, live_config, tmp_path / "run", robot, np.eye(4))
    assert events == ["robot closed", "camera closed", "backend closed"]


def test_synthetic_capture_cannot_execute(scene, live_config, tmp_path):
    capture = scene[0]
    backend = AnyGraspBackend(Records([dict(rotation=np.eye(3), translation=[.3, 0, .3],
                                           depth=0, width=.04, score=.9)]))
    result, _ = infer_snapshot(capture, None, backend, live_config, tmp_path)
    robot = SimpleNamespace(execute=lambda p: pytest.fail("synthetic motion"))
    with pytest.raises(ValueError, match="live capture"):
        execute_snapshot(capture, result.plans[0], robot, tmp_path)


def test_execute_cli_missing_license_never_constructs_robot(monkeypatch, tmp_path):
    from rebot_grasp import bootstrap, cli
    config = Path(__file__).resolve().parents[1] / "config/anygrasp.yaml"
    monkeypatch.setattr(commands, "make_backend", lambda config: (_ for _ in ()).throw(FileNotFoundError("license missing")))
    monkeypatch.setattr(bootstrap, "make_robot", lambda config: pytest.fail("robot constructed"))
    # Use a temporary hand-eye file so the test also works in a clean checkout.
    import yaml
    import shutil
    from rebot_grasp.config import load_profile
    raw = load_profile(config)
    matrix = tmp_path / "hand_eye.npz"
    np.savez(matrix, T_result=np.eye(4), mode=["eye_in_hand"], n_samples=[18])
    raw["calibration"]["hand_eye_path"] = str(matrix)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    shutil.copytree(config.parent / "backends", tmp_path / "backends")
    assert cli.main(["anygrasp-live", "--config", str(path), "--execute",
                     "--output", str(tmp_path / "run")]) == 1


def test_execute_failure_retains_attempt_record(scene, live_config, tmp_path):
    capture = Capture(scene[0].frame, np.eye(4))
    backend = AnyGraspBackend(Records([dict(rotation=np.eye(3), translation=[.3, 0, .3],
                                           depth=0, width=.04, score=.9)]))
    result, _ = infer_snapshot(capture, None, backend, live_config, tmp_path)
    def fail(plan):
        raise RuntimeError("CAN failure")
    with pytest.raises(RuntimeError, match="CAN failure"):
        execute_snapshot(capture, result.plans[0], SimpleNamespace(execute=fail), tmp_path)
    report = json.loads((tmp_path / "execution.json").read_text())
    assert report["execution_attempted"]
    assert not report["call_completed"] and not report["object_held"]
    assert report["error"] == "RuntimeError: CAN failure"


@pytest.mark.parametrize("diagnose", [False, True])
def test_live_g_always_captures_new_frame_and_uses_live_session(monkeypatch, scene, live_config, tmp_path, diagnose):
    from rebot_grasp.pipeline.anygrasp import run_live
    import cv2
    import itertools
    events = []
    frames = itertools.count()
    keys = iter([ord('g'), ord('g'), ord('q')])
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: None)
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda delay: next(keys) if delay == 20 else -1)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *a: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    camera = SimpleNamespace(open=lambda: events.append('camera opened'),
                             capture=lambda: replace(scene[0].frame, frame_id=f'frame-{next(frames)}'),
                             close=lambda: events.append('camera closed'))
    records = Records([dict(rotation=np.eye(3), translation=[.3, 0, .3], depth=0, width=.04, score=.9)])
    records.start = lambda: events.append('model loaded')
    records.close = lambda: events.append('model closed')
    backend = AnyGraspBackend(records)
    robot = SimpleNamespace(start=lambda: events.append('robot started'),
                            close=lambda: events.append('robot closed'),
                            tcp_pose=lambda: np.eye(4), reachable=lambda pose: True,
                            execute=lambda plan: events.append(plan.candidate.frame_id) or True)
    output = tmp_path / 'run'
    if diagnose:
        robot.gripper = SimpleNamespace(open_gripper=lambda: events.append("opening"),
                                        confirm_open=lambda: events.append("confirmed"))
        robot.execute = lambda plan: pytest.fail("diagnostic dispatched a grasp")
    run_live(camera, backend, None, live_config, output, robot, np.eye(4), diagnose_gripper=diagnose)
    if diagnose:
        assert events.count("opening") == events.count("confirmed") == 1
        assert events[-3:] == ["robot closed", "camera closed", "model closed"]
        assert not list(output.glob("snapshot-*/execution.json"))
        assert json.loads((output / "session.json").read_text())["execute"] is False
        return
    reports = [json.loads(p.read_text()) for p in sorted(output.glob('snapshot-*/execution.json'))]
    assert len(reports) == 2
    assert reports[0]['frame_id'] != reports[1]['frame_id']
    assert all(r['object_held'] for r in reports)
    assert events[:3] == ['model loaded', 'camera opened', 'robot started']
    assert events[-3:] == ['robot closed', 'camera closed', 'model closed']


def test_grasp_preview_projects_best_jaw_axis_and_clears_missing_target(scene, live_config):
    from rebot_grasp.pipeline.anygrasp import LiveGraspPreview

    capture, target = scene
    other = replace(target, label="cup", confidence=1.0, mask=~target.mask)
    detections = iter([[other, target], [other]])
    records = Records([dict(rotation=np.eye(3), translation=[0.04, 0, .45], depth=0, width=.04, score=.4),
                       dict(rotation=np.eye(3), translation=[0, 0, .45], depth=0, width=.04, score=.9)])
    config = replace(live_config, full_scene=False, target_class=target.label)
    worker = LiveGraspPreview(SimpleNamespace(detect=lambda _: next(detections)), AnyGraspBackend(records), config)
    try:
        found = worker._process(capture.frame)
        missing = worker._process(capture.frame)
    finally:
        worker.close()

    y, x = np.nonzero(target.mask)
    np.testing.assert_array_equal(found.image[y.min(), x.min()], [40, 230, 40])
    np.testing.assert_array_equal(missing.image[y.min(), x.min()], capture.frame.color_bgr[y.min(), x.min()])
    # Projection at fx=fy=180: center (80,60), half-opening 8 px along image Y.
    np.testing.assert_array_equal(found.image[60, 80], [0, 0, 255])
    np.testing.assert_array_equal(found.image[53, 80], [255, 255, 255])
    np.testing.assert_array_equal(found.image[60, 96], capture.frame.color_bgr[60, 96])
    # The remaining cup still gets its own preview; the missing active target cannot execute.
    assert missing.result.status == "target_not_found" and not missing.result.candidates
    assert len(records.calls) == 3 and not found.result.plans
    assert not np.array_equal(records.calls[0][1]["region_steering"], records.calls[1][1]["region_steering"])
    np.testing.assert_array_equal(records.calls[0][1]["region_steering"], records.calls[2][1]["region_steering"])
    assert not np.shares_memory(found.image, capture.frame.color_bgr)
