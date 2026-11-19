from rebot_grasp.pipeline import commands
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from rebot_grasp import cli
from rebot_grasp.cameras.offline import save_recording


def synthetic_camera_profile(monkeypatch):
    # Synthetic 160x120 images do not use the physical 640x480 mount silhouette.
    loader = commands.load_config
    def load(*args, **kwargs):
        config = loader(*args, **kwargs)
        return replace(config, backend_options={**config.backend_options, 'gripper_exclusion': None})
    monkeypatch.setattr(commands, 'load_config', load)


def test_camera_check_exports_aligned_mask_without_grasp_sdk_or_camera(tmp_path, monkeypatch, scene):
    synthetic_camera_profile(monkeypatch)
    capture, target = scene
    recording = tmp_path / 'input.npz'
    save_recording(recording, capture)
    detector = SimpleNamespace(start=lambda: None, detect=lambda frame: [target])
    monkeypatch.setattr(commands, 'make_detector', lambda config: detector)
    def forbidden(*args):
        raise AssertionError('Unexpected device or grasp SDK')
    monkeypatch.setattr(commands, 'make_backend', forbidden)
    monkeypatch.setattr(commands, 'make_camera', forbidden)
    output = tmp_path / 'check'
    assert cli.main(['camera-check', '--input', str(recording), '--output', str(output),
                     '--target-class', target.label]) == 0
    report = json.loads((output / 'inspection.json').read_text())
    cloud = np.load(output / 'pointcloud.npz')
    assert report['selected_label'] == target.label
    assert cloud['points_m'].shape == cloud['colors_rgb'].shape
    assert cloud['target_mask'].shape == (len(cloud['points_m']),)
    assert cloud['target_mask'].sum() == report['selected_point_count'] > 0
    assert (output / 'segmentation.png').is_file()
    assert (output / 'mask-00.png').is_file()


def test_camera_check_uses_configured_target_not_highest_score(tmp_path, monkeypatch, scene):
    synthetic_camera_profile(monkeypatch)
    from rebot_grasp.core.types import Target
    capture, target = scene
    recording = tmp_path / 'input.npz'
    save_recording(recording, capture)
    targets = [Target('tool', target.mask, .9), Target('yellow banana', target.mask, .4)]
    monkeypatch.setattr(commands, 'make_detector', lambda config: SimpleNamespace(
        start=lambda: None, detect=lambda frame: targets))
    output = tmp_path / 'check'
    assert cli.main(['camera-check', '--input', str(recording), '--output', str(output)]) == 0
    report = json.loads((output / 'inspection.json').read_text())
    assert report['selected_label'] == 'yellow banana'
    assert report['outlier_filter'] == {'nb_neighbors': 20, 'std_ratio': 2.0}
    raw = np.load(output / 'pointcloud_raw.npz')
    filtered = np.load(output / 'pointcloud.npz')
    assert len(raw['points_m']) - len(filtered['points_m']) == report['removed_point_count']


def test_camera_check_quality_failure_saves_diagnostics_and_returns_two(tmp_path, monkeypatch, scene):
    from rebot_grasp.core.types import Target, Capture
    synthetic_camera_profile(monkeypatch)
    capture, target = scene
    depth = capture.frame.depth_m.copy()
    depth[target.mask] = 0
    save_recording(tmp_path / 'input.npz', Capture(replace(capture.frame, depth_m=depth)))
    monkeypatch.setattr(commands, 'make_detector', lambda config: SimpleNamespace(
        start=lambda: None, detect=lambda frame: [Target('yellow banana', target.mask)]))
    output = tmp_path / 'check'
    assert cli.main(['camera-check', '--input', str(tmp_path / 'input.npz'), '--output', str(output)]) == 2
    report = json.loads((output / 'inspection.json').read_text())
    assert not report['quality']['passed']
    assert 'insufficient_valid_depth' in report['quality']['reasons']
    assert (output / 'preprocessing.png').is_file()
    assert (output / 'eligible_target.png').is_file()
