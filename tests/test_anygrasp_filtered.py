import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rebot_grasp.config import load_config
from rebot_grasp.core.types import Capture
from rebot_grasp.grasping.backends.anygrasp_filtered import FilteredAnyGraspBackend
from rebot_grasp.pipeline.anygrasp_filtered import ComparisonPreview, infer_snapshot


class DetectorSDK:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.inputs = []

    def predict(self, points, options):
        self.inputs.append((len(points), options["region_steering"].copy()))
        return next(self.batches)


def grasp(x, score):
    return dict(rotation=np.eye(3), translation=[x, 0., .45], depth=0., width=.04, score=score)


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "config/anygrasp_filtered.yaml"
    return replace(load_config(path), target_class="synthetic-block")


def test_same_model_two_branches_raw_never_selected(scene, config, tmp_path):
    frame, target = scene[0].frame, scene[1]
    capture = Capture(frame, np.eye(4))
    sdk = DetectorSDK([[grasp(.08, 1.)], [grasp(.01, .99), grasp(.04, .8)]])
    backend = FilteredAnyGraspBackend(sdk, config.filters, insertion_depth_m=.005,
                                      workspace_camera=dict(min_m=[0., -.2, .1], max_m=[.2, .2, .5]))
    detections = []
    def detect(frame):
        detections.append(frame.frame_id)
        return [target]
    robot = SimpleNamespace(reachable=lambda pose: pose[0, 3] > -.05)
    result, selected = infer_snapshot(capture, SimpleNamespace(detect=detect), backend, config, tmp_path, robot)
    assert len(sdk.inputs) == 2 and detections == [frame.frame_id]
    assert 0 < sdk.inputs[1][1].sum() < sdk.inputs[0][1].sum()
    assert selected.candidate.score == .8 and all(c.score != 1. for c in result.candidates)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["raw"]["score"] == 1. and report["filtered"]["score"] == .8
    assert report["filtered_selected_for_execution"]
    assert report["delta"]["distance_m"] == pytest.approx(.04)
    assert report["insertion_depth_m"] == .005


def test_quality_rejection_and_missing_target_clear_comparison(scene, config):
    frame, target = scene[0].frame, scene[1]
    sdk = DetectorSDK([[grasp(.04, .95)]])
    backend = FilteredAnyGraspBackend(sdk, config.filters, pointcloud_quality=dict(min_target_points=10000))
    batches = iter([[target], []])
    preview = ComparisonPreview(SimpleNamespace(detect=lambda _: next(batches)), backend, config)
    try:
        rejected = preview._process(frame)
        missing = preview._process(replace(frame, frame_id="absent"))
    finally:
        preview.close()
    assert rejected.result.status == "pointcloud_rejected" and not rejected.result.candidates
    assert rejected.backend_view.reference is not None and len(sdk.inputs) == 1
    assert missing.result.status == "target_not_found" and not missing.result.plans
    assert missing.backend_view.reference is None and missing.backend_view.last_prepared is None


def test_desktop_objects_visible_in_both_panels_even_without_grasp_target(scene, config):
    from rebot_grasp.core.types import Target

    frame = scene[0].frame
    objects = []
    for label, x in (("pen", 10), ("orange", 50), ("yellow banana", 100)):
        mask = np.zeros(frame.depth_m.shape, bool)
        mask[65:75, x:x + 8] = True
        objects.append(Target(label, mask, .8))
    frames = iter([objects, objects[:2]])
    # A distinct projected center for each instance, once per raw/filtered input.
    sdk = DetectorSDK([[grasp((u - 80) * .45 / 180, .8)]
                       for u in (14, 14, 54, 54, 104, 104, 14, 14, 54, 54)])
    backend = FilteredAnyGraspBackend(sdk, config.filters)
    worker = ComparisonPreview(SimpleNamespace(detect=lambda _: next(frames)), backend,
                               replace(config, target_class="yellow banana"))
    try:
        found = worker._process(frame)
        missing = worker._process(replace(frame, frame_id="no-banana"))
    finally:
        worker.close()
    for offset in (0, frame.color_bgr.shape[1]):
        for x in (10, 50):
            np.testing.assert_array_equal(found.image[74, x + offset], [230, 160, 40])
            np.testing.assert_array_equal(missing.image[74, x + offset], [230, 160, 40])
        np.testing.assert_array_equal(found.image[74, 100 + offset], [40, 230, 40])
        for u in (14, 54):
            np.testing.assert_array_equal(found.image[54, u + offset], [0, 205, 255])
            np.testing.assert_array_equal(missing.image[54, u + offset], [0, 205, 255])
    assert not missing.result.candidates and len(sdk.inputs) == 10


def test_keyboard_switch_dispatches_only_fresh_pen_then_orange(monkeypatch, scene, config, tmp_path):
    import cv2
    from itertools import count

    from rebot_grasp.core.types import Target
    from rebot_grasp.pipeline.anygrasp_filtered import run_live

    frame = scene[0].frame
    seq = count()
    camera = SimpleNamespace(open=lambda: None, close=lambda: None,
                             capture=lambda: replace(frame, frame_id=str(next(seq))))
    objects = []
    for label, x in (("pen", 28), ("orange", 80), ("yellow banana", 124)):
        mask = np.zeros(frame.depth_m.shape, bool)
        mask[55:65, x:x + 8] = True
        objects.append(Target(label, mask, .9))

    class SDK:
        def start(self):
            pass

        def predict(self, points, options):
            center = points[options["region_steering"]].mean(axis=0)
            return [dict(rotation=np.eye(3), translation=center, depth=0., width=.03, score=.9)]

    actions = []
    robot = SimpleNamespace(start=lambda: None, close=lambda: None, tcp_pose=lambda: np.eye(4),
                            reachable=lambda pose: True, execute=lambda plan: actions.append(plan) or True)
    keys = iter(map(ord, "1g2gq"))
    for name in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys) if delay == 20 else -1)
    output = tmp_path / "grasp"
    run_live(camera, FilteredAnyGraspBackend(SDK(), config.filters),
             SimpleNamespace(start=lambda: None, detect=lambda _: objects),
             replace(config, target_class="yellow banana"), output, robot, np.eye(4), show_diagnostics=False)
    assert len(actions) == 2
    for folder, label, plan in zip(sorted(output.glob("snapshot-*")), ("pen", "orange"), actions):
        report = json.loads((folder / "result.json").read_text())
        assert report["target_class"] == label and report["frame_id"] == plan.candidate.frame_id
        with np.load(folder / "frame.npz") as recording:
            assert recording["target_label"].item() == label
    assert actions[0].candidate.T_camera_grasp[0, 3] < actions[1].candidate.T_camera_grasp[0, 3]
