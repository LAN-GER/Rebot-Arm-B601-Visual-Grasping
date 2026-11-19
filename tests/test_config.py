from pathlib import Path
import shutil

import pytest
import yaml

from rebot_grasp.config import load_config


def copied_config(tmp_path):
    origin = Path(__file__).resolve().parents[1] / "config"
    shutil.copytree(origin / "backends", tmp_path / "backends")
    raw = yaml.safe_load((origin / "default.yaml").read_text())
    return tmp_path / "default.yaml", raw


def test_sdk_paths_are_relative_to_backend_yaml_not_cwd(tmp_path, monkeypatch):
    path, raw = copied_config(tmp_path)
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir("/")
    config = load_config(path, backend="anygrasp")
    assert config.backend_options["sdk_root"] == str((tmp_path / "../sdk/anygrasp_sdk").resolve())


def test_default_config_projects_backend_settings_for_grasp_examples():
    from rebot_grasp.config import load_example_config

    path = Path(__file__).resolve().parents[1] / "config/default.yaml"
    config = load_config(path, backend="graspnet")
    settings = load_example_config(path)
    assert settings["camera"] == config.camera
    assert settings["robot"] == config.robot
    assert settings["graspnet"]["checkpoint"] == config.backend_options["checkpoint"]
    assert settings["graspnet"]["min_depth"] == config.backend_options["min_depth_m"]
    assert settings["grasp_pipeline"]["grasp"]["pregrasp_offset_m"] == config.planning["pregrasp_offset_m"]
    assert settings["grasp_pipeline"]["grasp"]["depth_quantile"] == load_config(path).backend_options["depth_quantile"]


@pytest.mark.parametrize("override", [
    {"full_scene": True, "target_class": "cup", "backend": "anygrasp"},
    {"filters": {"max_opening_m": -1}},
    {"planning": {"pregrasp_offset_m": float("nan")}},
    {"full_scene": "false"},
    {"unknown_key": 1},
])
def test_invalid_configuration_rejected_before_hardware_access(tmp_path, override):
    path, raw = copied_config(tmp_path)
    raw.update(override)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_live_profile_inherits_calibration_and_gripper_settings(tmp_path):
    path, base = copied_config(tmp_path)
    base["calibration"]["hand_eye_path"] = "calibration/replacement.npz"
    path.write_text(yaml.safe_dump(base))
    profile = tmp_path / "live.yaml"
    profile.write_text("use_defaults: true\nrobot:\n  gripper:\n    rs:\n      default_force: 0.25\n")
    live = load_config(profile, "anygrasp")
    assert live.calibration == base["calibration"]
    assert live.robot["gripper"]["rs"]["default_force"] == 0.25
    assert live.robot["gripper"]["rs"]["angle_open"] == base["robot"]["gripper"]["rs"]["angle_open"]
    assert live.robot["ready_pose"] == base["robot"]["ready_pose"]
