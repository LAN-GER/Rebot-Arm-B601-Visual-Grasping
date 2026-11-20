import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from rebot_grasp.cli import main


def test_demo_then_replay_without_missing_transform_fallback(tmp_path):
    assert main(["demo", "--output", str(tmp_path / "demo")]) == 0
    payload = json.loads((tmp_path / "demo/result.json").read_text())
    assert payload["synthetic"] and not payload["executed"]
    assert len(payload["plans"]) == 1
    assert (tmp_path / "demo/preview.png").is_file()
    assert main(["preview", "--input", str(tmp_path / "demo/frame.npz"), "--output", str(tmp_path / "replay")]) == 0
    replay = json.loads((tmp_path / "replay/result.json").read_text())
    assert replay["status"] == "camera_preview_only_missing_transforms"
    assert replay["synthetic"]  # Replaying a synthetic capture preserves provenance.
    assert replay["plans"] == []


def test_recording_in_millimeters_is_not_silently_accepted(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, color_bgr=np.zeros((2, 2, 3), np.uint8), depth_mm=np.ones((2, 2)), K=np.eye(3))
    assert main(["preview", "--input", str(path), "--output", str(tmp_path / "out")]) == 1


def test_graspnet_replay_starts_backend_before_detection(monkeypatch, tmp_path):
    from rebot_grasp.cameras.offline import save_recording
    from rebot_grasp.cameras.synthetic import demo_capture
    from rebot_grasp.pipeline import commands

    capture, target = demo_capture()
    recording = tmp_path / "frame.npz"
    save_recording(recording, capture, target)
    calls = []

    def detect(_frame):
        calls.append("detect")
        return [target]

    def predict(_frame, _target):
        calls.append("predict")
        return []

    backend = SimpleNamespace(start=lambda: calls.append("start"), predict=predict)
    monkeypatch.setattr(commands, "make_backend", lambda _config: backend)
    monkeypatch.setattr(commands, "make_detector", lambda _config: SimpleNamespace(detect=detect))
    assert main(["preview", "--backend", "graspnet", "--detect", "--no-image",
                 "--input", str(recording), "--output", str(tmp_path / "result")]) == 2
    assert calls == ["start", "detect", "predict"]


def test_importing_all_new_modules_does_not_import_optional_sdks():
    script = '''
import importlib, pkgutil, sys
import rebot_grasp
def import_layers(package):
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        if info.name == 'rebot_grasp.__main__':
            continue
        module = importlib.import_module(info.name)
        if info.ispkg:
            import_layers(module)
import_layers(rebot_grasp)
for name in ['torch', 'gsnet', 'ultralytics', 'open3d', 'pyrealsense2', 'pyorbbecsdk', 'reBotArm_control_py', 'cv2']:
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cli_runs_outside_repo_with_explicit_config(tmp_path):
    config = Path(__file__).resolve().parents[1] / "config/default.yaml"
    process = subprocess.run([sys.executable, "-m", "rebot_grasp", "demo", "--config", str(config), "--output", str(tmp_path / "out")], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert (tmp_path / "out/result.json").is_file()
