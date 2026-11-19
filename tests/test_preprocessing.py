from dataclasses import replace
import json

import numpy as np
import pytest

from rebot_grasp.core.types import RGBDFrame, Target, Capture
from rebot_grasp.grasping.backends.anygrasp import AnyGraspBackend
from rebot_grasp.perception.preprocessing import prepare_cloud, validate_preprocessing
from rebot_grasp.pipeline.planning import plan_capture
from rebot_grasp.visualization.export import save_result


def frame(depth=None):
    depth = np.full((10, 10), .3, np.float32) if depth is None else depth
    color = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
    return RGBDFrame(color, depth, [[100, 0, 5], [0, 100, 5], [0, 0, 1]], 'quality-test', 0)


PROFILE = dict(reference_resolution=[10, 10], min_depth_m=.05, max_depth_m=.19,
               polygons_uv=[[[0, .55], [1, .55], [1, 1], [0, 1]]])
WORKSPACE = dict(min_m=[-.1, -.1, .2], max_m=[.1, .1, .4])


class Detector:
    def __init__(self, records=()):
        self.calls = []
        self.records = records

    def predict(self, points, options):
        self.calls.append((points, options))
        return self.records


def test_gripper_depth_gate_keeps_far_object_in_same_pixels_and_alignment():
    d = np.full((10, 10), .3, np.float32)
    d[7:, :5] = .15
    f = frame(d)
    t = Target('banana', np.ones((10, 10), bool))
    result = prepare_cloud(f, t, gripper_exclusion=PROFILE)
    assert result.report['counts']['gripper_removed_points'] == 15
    assert len(result.cloud.points) == 85
    assert result.cloud.target_mask.all()
    np.testing.assert_allclose(result.cloud.colors_rgb, f.color_bgr[~result.gripper_mask][:, ::-1] / 255)
    assert not result.gripper_mask[7:, 5:].any()


def test_workspace_steers_without_removing_collision_context_in_full_scene():
    d = np.full((10, 10), .3, np.float32)
    d[:, :4] = .5
    result = prepare_cloud(frame(d), None, workspace_camera=WORKSPACE)
    assert len(result.cloud.points) == 100
    assert result.cloud.target_mask.sum() == 60
    assert (result.cloud.points[~result.cloud.target_mask, 2] == .5).all()
    assert result.report['quality']['passed']


@pytest.mark.parametrize('change,limits,reason', [
    ('holes', {'min_valid_depth_ratio': .6}, 'insufficient_valid_depth'),
    ('normal', {'min_target_points': 101}, 'insufficient_target_points'),
    ('normal', {'min_scene_points': 101}, 'insufficient_scene_points'),
    ('outside', {'min_retained_ratio': .5}, 'excessive_region_removal'),
])
def test_quality_rejections_never_call_sdk(change, limits, reason):
    d = np.full((10, 10), .3, np.float32)
    if change == 'holes':
        d[:6] = 0
    if change == 'outside':
        d[:8] = .5
    detector = Detector()
    backend = AnyGraspBackend(detector, workspace_camera=WORKSPACE, pointcloud_quality=limits)
    assert backend.predict(frame(d), Target('banana', np.ones((10, 10), bool))) == []
    assert reason in backend.last_preprocessing_report['quality']['reasons']
    assert detector.calls == []


def test_all_target_removed_by_gripper_never_becomes_scene_inference():
    d = np.full((10, 10), .3, np.float32)
    d[7:] = .15
    t = Target('banana', d < .2)
    detector = Detector()
    backend = AnyGraspBackend(detector, gripper_exclusion=PROFILE)
    assert backend.predict(frame(d), t) == []
    assert detector.calls == []
    assert backend.last_preprocessing_report['counts']['gripper_removed_target_points'] == 30


def test_empty_workspace_full_scene_never_disables_steering():
    detector = Detector()
    backend = AnyGraspBackend(detector, workspace_camera=WORKSPACE)
    assert backend.predict(frame(np.full((10, 10), .5, np.float32)), None) == []
    assert detector.calls == []


def test_excessive_outliers_rejected_and_report_is_serializable():
    d = np.full((10, 10), .3, np.float32)
    d[0, 0] = .9
    t = Target('banana', np.ones((10, 10), bool))
    r = prepare_cloud(frame(d), t, outlier_filter={'nb_neighbors': 5, 'std_ratio': 2.},
                      pointcloud_quality={'max_outlier_ratio': 0.})
    assert 'excessive_outliers' in r.report['quality']['reasons']
    assert r.report['counts']['outlier_removed_target_points'] >= 1
    json.dumps(r.report, allow_nan=False)


def test_missing_target_inspection_rejected_even_with_healthy_scene():
    r = prepare_cloud(frame(), None, target_required=True)
    assert 'target_not_found' in r.report['quality']['reasons']
    assert r.report['counts']['raw_target_points'] == 0
    assert r.report['quality']['metrics']['target_points'] == 0
    assert not r.cloud.target_mask.any()


def test_gripper_profile_rejects_resolution_mismatch():
    with pytest.raises(ValueError, match='resolution mismatch'):
        prepare_cloud(frame(), None, gripper_exclusion={**PROFILE, 'reference_resolution': [640, 480]})


@pytest.mark.parametrize('settings', [
    {'workspace_camera': {'min_m': [0, 0, 0], 'max_m': [0, 1, 1]}},
    {'workspace_camera': {'min_m': [0, 0, 0], 'max_m': [1, 1, float('inf')]}},
    {'gripper_exclusion': {**PROFILE, 'polygons_uv': [[[0, 0], [1, 1], [2, 1]]]}},
    {'gripper_exclusion': {**PROFILE, 'polygons_uv': [[[0, 0], [.5, .5], [1, 1]]]}},
    {'pointcloud_quality': {'min_valid_depth_ratio': 1.1}},
    {'pointcloud_quality': {'min_target_points': True}},
    {'pointcloud_quality': {'unknown': 1}},
    {'outlier_filter': {'nb_neighbors': 1}},
])
def test_bad_settings_fail_before_hardware(settings):
    with pytest.raises(ValueError):
        validate_preprocessing(**settings)


def test_final_sdk_candidate_outside_workspace_is_rejected():
    detector = Detector([dict(rotation=np.eye(3), translation=[0, 0, .8],
                              depth=.02, width=.04, score=.9)])
    backend = AnyGraspBackend(detector, workspace_camera=WORKSPACE)
    assert backend.predict(frame(), None) == []
    assert len(detector.calls) == 1
    assert detector.calls[0][1]['region_steering'].all()


def test_planning_exports_quality_failure_instead_of_no_grasp(config, tmp_path):
    detector = Detector()
    backend = AnyGraspBackend(detector, pointcloud_quality={'min_target_points': 101})
    config = replace(config, full_scene=False, target_class='banana')
    capture = Capture(frame())
    result = plan_capture(capture, [Target('banana', np.ones((10, 10), bool))], backend, config)
    assert result.status == 'pointcloud_rejected'
    assert not result.plans and not detector.calls
    path = save_result(tmp_path, capture, result)
    report = json.loads(path.read_text())
    assert report['pointcloud_report']['quality']['reasons'] == ['insufficient_target_points']


def test_snapshot_quality_failure_does_not_query_ik_or_execute(config, tmp_path):
    from types import SimpleNamespace
    from rebot_grasp.pipeline.anygrasp import infer_snapshot
    f = frame()
    target = Target('banana', np.ones((10, 10), bool))
    detector = SimpleNamespace(detect=lambda frame: [target])
    backend = AnyGraspBackend(Detector(), pointcloud_quality={'min_target_points': 101})
    def forbidden(*args):
        pytest.fail('Robot accessed after quality rejection')
    robot = SimpleNamespace(reachable=forbidden, execute=forbidden)
    result, plan = infer_snapshot(Capture(f, np.eye(4)), detector, backend,
                                  replace(config, target_class='banana', full_scene=False), tmp_path, robot)
    assert result.status == 'pointcloud_rejected' and plan is None
    assert (tmp_path / 'result.json').is_file()
