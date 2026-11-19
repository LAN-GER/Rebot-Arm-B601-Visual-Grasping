from rebot_grasp.pipeline import commands
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rebot_grasp import cli
from rebot_grasp.core.types import Capture
from rebot_grasp.config import load_config
from rebot_grasp.cameras.offline import save_recording
from rebot_grasp.grasping.backends.obb_filtered import FilteredOBBBackend, estimate_contour_pose
from rebot_grasp.geometry.transforms import pose6d
from rebot_grasp.geometry.tool_frames import canonicalize_gripper_plan
from rebot_grasp.pipeline.planning import plan_capture
from rebot_grasp.pipeline.obb_filtered import infer_snapshot, run_live


@pytest.fixture
def filtered_config():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'config/obb_filtered.yaml')
    return replace(cfg, target_class='synthetic-block',
                   backend_options={**cfg.backend_options, 'gripper_exclusion': None})


def test_no_filters_preserves_contour_geometry_and_base_pose(scene, filtered_config):
    from rebot_grasp.geometry.grasp_frames import transform_grasp_pose_to_base
    capture, target = scene
    backend = FilteredOBBBackend()
    candidate, = backend.predict(capture.frame, target)
    reference = estimate_contour_pose(capture.frame, target, capture.frame.depth_m, .5)
    np.testing.assert_allclose(candidate.T_camera_grasp[:3, 3], reference.position)
    np.testing.assert_allclose(candidate.T_camera_grasp[:3, :3], reference.tcp_rotation)
    assert candidate.width_m == reference.jaw_width_m
    plan = canonicalize_gripper_plan(plan_capture(capture, [target], backend, filtered_config).plans[0])
    grasp, pre = transform_grasp_pose_to_base(reference.position, reference.tcp_rotation,
                                             capture.T_base_camera, .08)
    np.testing.assert_allclose(pose6d(plan.T_base_grasp), grasp, atol=1e-6)
    np.testing.assert_allclose(pose6d(plan.T_base_pregrasp), pre, atol=1e-6)


def test_workspace_filtered_pixels_change_depth_in_actual_obb_input(scene):
    capture, target = scene
    depth = capture.frame.depth_m.copy()
    ys, xs = np.where(target.mask)
    depth[ys[:len(ys)*3//4], xs[:len(xs)*3//4]] = .9
    frame = replace(capture.frame, depth_m=depth)
    backend = FilteredOBBBackend(workspace_camera={'min_m': [-1, -1, .1], 'max_m': [1, 1, .7]})
    result, = backend.predict(frame, target)
    assert backend.last_comparison['raw']['position'][2] == pytest.approx(.9)
    assert result.T_camera_grasp[2, 3] == pytest.approx(.45)
    assert backend.last_comparison['filtered']['valid_depth_pixels'] == 192
    assert not backend.last_eligible_mask[ys[0], xs[0]]
    comparison = backend.last_comparison
    assert comparison['raw']['center_px'] == comparison['filtered']['center_px']
    assert comparison['raw']['angle_deg'] == comparison['filtered']['angle_deg']
    np.testing.assert_array_equal(comparison['raw']['rect_points'], comparison['filtered']['rect_points'])
    np.testing.assert_array_equal(comparison['raw']['short_edge_points'], comparison['filtered']['short_edge_points'])
    # Metric width scales with the filtered depth; its pixel span stays fixed.
    assert comparison['filtered']['jaw_width_m'] / comparison['raw']['jaw_width_m'] == pytest.approx(.45/.9)
    assert comparison['geometry_source'] == 'full_2d_segmentation'
    assert comparison['insertion_depth_m'] == 0


def test_depth_holes_do_not_shift_full_silhouette_center_or_direction(scene):
    capture, target = scene
    depth = capture.frame.depth_m.copy()
    depth[36:66, 72:88] = 0
    modified = replace(capture.frame, depth_m=depth)
    backend = FilteredOBBBackend()
    candidate, = backend.predict(modified, target)
    reference = estimate_contour_pose(capture.frame, target, capture.frame.depth_m, .5)
    filtered = backend.last_comparison['filtered']
    assert filtered['center_px'] == reference.center_px
    assert filtered['angle_deg'] == reference.angle_deg
    np.testing.assert_allclose(candidate.T_camera_grasp[:3, 3], reference.position)
    np.testing.assert_allclose(candidate.T_camera_grasp[:3, :3], reference.tcp_rotation)
    assert filtered['valid_depth_pixels'] == 18*16


def test_insertion_applied_once_along_tool_axis_and_pregrasp_offset_preserved(scene, filtered_config):
    capture, target = scene
    baseline = plan_capture(capture, [target], FilteredOBBBackend(), filtered_config).plans[0]
    backend = FilteredOBBBackend(insertion_depth_m=.005)
    changed = plan_capture(capture, [target], backend, filtered_config).plans[0]
    expected = .005 * baseline.T_base_grasp[:3, 0]
    np.testing.assert_allclose(changed.T_base_grasp[:3, 3] - baseline.T_base_grasp[:3, 3], expected, atol=1e-7)
    np.testing.assert_allclose(changed.T_base_pregrasp[:3, 3] - baseline.T_base_pregrasp[:3, 3], expected, atol=1e-7)
    assert np.linalg.norm(changed.T_base_grasp[:3, 3] - changed.T_base_pregrasp[:3, 3]) == pytest.approx(.08)
    np.testing.assert_array_equal(changed.T_base_grasp[:3, :3], baseline.T_base_grasp[:3, :3])
    assert changed.candidate.width_m == baseline.candidate.width_m
    assert backend.last_comparison['insertion_depth_m'] == .005
    np.testing.assert_allclose(backend.last_comparison['candidate_position_m'], changed.candidate.T_camera_grasp[:3, 3])


def test_inserted_candidate_is_checked_against_workspace(scene):
    capture, target = scene
    options = dict(workspace_camera={'min_m': [-1, -1, .1], 'max_m': [1, 1, .452]})
    assert FilteredOBBBackend(**options).predict(capture.frame, target)
    backend = FilteredOBBBackend(**options, insertion_depth_m=.005)
    assert backend.predict(capture.frame, target) == []
    assert backend.last_comparison['status'] == 'inserted_center_outside_workspace'


@pytest.mark.parametrize('value', [-.005, float('nan'), float('inf'), True])
def test_invalid_insertion_rejected(value):
    with pytest.raises(ValueError, match='insertion_depth_m'):
        FilteredOBBBackend(insertion_depth_m=value)


def test_quality_rejection_never_falls_back_to_raw_grasp(scene, filtered_config, tmp_path):
    capture, target = scene
    backend = FilteredOBBBackend(pointcloud_quality={'min_target_points': 10000})
    detector = SimpleNamespace(start=lambda: None, detect=lambda frame: [target])
    robot = SimpleNamespace(reachable=lambda p: pytest.fail('IK after rejection'))
    result, plan = infer_snapshot(capture, detector, backend, filtered_config, tmp_path, robot)
    assert result.status == 'pointcloud_rejected' and not result.candidates and plan is None
    comparison = json.loads((tmp_path / 'comparison.json').read_text())
    assert comparison['raw']['position'] is not None
    assert comparison['filtered'] is None
    assert not (tmp_path / 'execution.json').exists()
    assert (tmp_path / 'pointcloud.npz').is_file()


def test_empty_filter_region_cannot_use_unfiltered_center_depth_fallback(scene):
    capture, target = scene
    backend = FilteredOBBBackend(workspace_camera={'min_m': [-1, -1, .8], 'max_m': [1, 1, 1.]})
    assert backend.predict(capture.frame, target) == []
    assert not backend.last_eligible_mask.any()
    assert backend.last_comparison['filtered'] is None


def test_missing_target_does_not_export_previous_frame_comparison(scene, filtered_config, tmp_path):
    capture, target = scene
    backend = FilteredOBBBackend()
    infer_snapshot(capture, SimpleNamespace(detect=lambda frame: [target]), backend,
                   filtered_config, tmp_path / 'one')
    new = Capture(replace(capture.frame, frame_id='new-frame'))
    result, plan = infer_snapshot(new, SimpleNamespace(detect=lambda frame: []), backend,
                                  filtered_config, tmp_path / 'two')
    record = json.loads((tmp_path / 'two/comparison.json').read_text())
    assert result.status == 'target_not_found' and plan is None
    assert record['frame_id'] == 'new-frame' and record['raw'] is None
    assert not (tmp_path / 'two/pointcloud.npz').exists()


def test_offline_cli_never_constructs_camera_robot_or_anygrasp(scene, filtered_config, tmp_path, monkeypatch):
    from rebot_grasp import bootstrap
    capture, target = scene
    save_recording(tmp_path / 'input.npz', capture)
    monkeypatch.setattr(commands, 'load_config', lambda *args: filtered_config)
    monkeypatch.setattr(commands, 'make_detector', lambda config: SimpleNamespace(
        start=lambda: None, detect=lambda frame: [target]))
    def forbidden(*args):
        pytest.fail('Hardware constructed during offline replay')
    monkeypatch.setattr(commands, 'make_camera', forbidden)
    monkeypatch.setattr(bootstrap, 'make_robot', forbidden)
    assert cli.main(['obb-filtered', '--input', str(tmp_path / 'input.npz'),
                     '--output', str(tmp_path / 'run'), '--insertion-depth-mm', '5']) == 0
    assert (tmp_path / 'run/comparison.png').is_file()
    assert json.loads((tmp_path / 'run/session.json').read_text())['execute'] is False
    assert json.loads((tmp_path / 'run/session.json').read_text())['config']['backend_options']['insertion_depth_m'] == .005
    assert json.loads((tmp_path / 'run/comparison.json').read_text())['insertion_depth_m'] == .005


def test_filtered_live_fresh_snapshots_and_only_filtered_plan_executed(scene, filtered_config, tmp_path, monkeypatch):
    import cv2
    import itertools
    capture, target = scene
    keys = iter([ord('g'), ord('r'), ord('g'), ord('q')])
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: None)
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda delay: next(keys) if delay == 20 else -1)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *a: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    count = itertools.count()
    calls = []
    robot = SimpleNamespace(start=lambda: None, close=lambda: None,
                             tcp_pose=lambda: capture.T_base_camera,
                             reachable=lambda pose: True,
                             execute=lambda plan: calls.append(plan) or False)
    camera = SimpleNamespace(open=lambda: None, close=lambda: None,
                              capture=lambda: replace(capture.frame, frame_id=str(next(count))))
    detector = SimpleNamespace(start=lambda: None, detect=lambda frame: [target])
    backend = FilteredOBBBackend()
    from rebot_grasp.visualization import obb_windows
    view_events = []
    observer = SimpleNamespace(start=lambda: view_events.append('start'),
                                poll=lambda: view_events.append('poll'),
                                update=lambda c, b, r: view_events.append(c.frame.frame_id),
                                close=lambda: view_events.append('close'))
    monkeypatch.setattr(obb_windows, 'OBBDiagnosticWindows', lambda: observer)
    run_live(camera, backend, detector, filtered_config, tmp_path / 'run', robot, np.eye(4))
    assert len(calls) == 2
    assert all(p.candidate.backend == 'obb_filtered' for p in calls)
    assert calls[0].candidate.frame_id != calls[1].candidate.frame_id
    assert view_events[0] == 'start' and view_events[-1] == 'close'
    assert all(p.candidate.frame_id in view_events for p in calls)
    for i in (1, 2):
        folder = tmp_path / f'run/snapshot-{i:03d}'
        report = json.loads((folder / 'execution.json').read_text())
        assert report['execution_attempted'] and report['call_completed']
        assert not report['object_held'] and report['field_success'] is None
        assert (folder / 'comparison.png').is_file()
        assert (folder / 'diagnostics.png').is_file()


def test_auxiliary_windows_use_snapshot_data_and_clear_missing_target(scene, filtered_config, tmp_path):
    from rebot_grasp.visualization.obb_windows import diagnostic_image, cloud_display_data
    capture, target = scene
    backend = FilteredOBBBackend()
    result, _ = infer_snapshot(capture, SimpleNamespace(detect=lambda f: [target]), backend,
                               filtered_config, tmp_path / 'one')
    original_colors = backend.last_prepared.cloud.colors_rgb.copy()
    points, colors, jaws = cloud_display_data(backend, result)
    assert points.shape == colors.shape
    assert len(jaws) == len(result.candidates) == 1
    np.testing.assert_allclose(colors[backend.last_prepared.cloud.target_mask],
                                np.tile([.1, 1., .1], (target.mask.sum(), 1)))
    np.testing.assert_array_equal(original_colors, backend.last_prepared.cloud.colors_rgb)
    h, w = capture.frame.depth_m.shape
    assert diagnostic_image(capture, backend, result).shape == (h + 64, w * 2, 3)
    result, _ = infer_snapshot(capture, SimpleNamespace(detect=lambda f: []), backend,
                               filtered_config, tmp_path / 'two')
    points, colors, jaws = cloud_display_data(backend, result)
    assert len(points) == len(colors) == len(jaws) == 0


def test_preview_live_never_executes_a_plan(scene, filtered_config, tmp_path, monkeypatch):
    import cv2
    capture, target = scene
    keys = iter([ord('g'), ord('q')])
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: None)
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda delay: next(keys) if delay == 20 else -1)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *a: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    camera = SimpleNamespace(open=lambda: None, close=lambda: None, capture=lambda: capture.frame)
    run_live(camera, FilteredOBBBackend(), SimpleNamespace(start=lambda: None, detect=lambda f: [target]),
             filtered_config, tmp_path / 'run', show_diagnostics=False)
    assert (tmp_path / 'run/snapshot-001/comparison.json').is_file()
    assert not (tmp_path / 'run/snapshot-001/execution.json').exists()
