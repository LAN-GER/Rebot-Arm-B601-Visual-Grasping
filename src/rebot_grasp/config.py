"""Load independent application/backend YAML files with explicit path semantics."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from rebot_grasp.core.types import checked_transform

BACKEND_KEYS = {
    "obb": {"depth_quantile", "insertion_depth_m", "min_depth_m", "max_depth_m"},
    "obb_filtered": {"depth_quantile", "min_depth_m", "max_depth_m", "insertion_depth_m", "outlier_filter",
                     "workspace_camera", "gripper_exclusion", "pointcloud_quality"},
    "anygrasp": {
        "sdk_root", "checkpoint", "max_gripper_width_m", "gripper_height_m", "timeout_s",
        "min_depth_m", "max_depth_m", "insertion_depth_m", "dense_grasp", "collision_detection",
        "approach_camera", "approach_thresh_rad", "outlier_filter",
        "workspace_camera", "gripper_exclusion", "pointcloud_quality",
    },
    "graspnet": {
        "sdk_root", "checkpoint", "num_point", "num_view", "collision_thresh", "voxel_size",
        "min_depth_m", "max_depth_m", "top_k", "target_margin_px", "target_expand_ratio",
    },
}

BACKEND_KEYS["anygrasp_filtered"] = BACKEND_KEYS["anygrasp"]


@dataclass(frozen=True)
class AppConfig:
    backend: str
    backend_options: dict
    target_class: str | None
    full_scene: bool
    filters: dict
    planning: dict
    yolo: dict
    camera: dict = field(default_factory=dict)
    robot: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)


def _mapping(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _merge_profile(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            value = _merge_profile(defaults[key], value)
        merged[key] = value
    return merged


def load_profile(path: str | Path) -> dict:
    """Read a complete profile or apply its overrides to the adjacent default.yaml."""
    path = Path(path).expanduser().resolve()
    profile = _mapping(path)
    use_defaults = profile.pop("use_defaults", False)
    if not isinstance(use_defaults, bool):
        raise ValueError("use_defaults must be a boolean")
    if not use_defaults:
        return profile
    defaults = path.with_name("default.yaml")
    if defaults == path:
        raise ValueError("default.yaml must contain the base configuration")
    return _merge_profile(_mapping(defaults), profile)


def _keys(value: dict, allowed: set, label: str):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")


def load_backend_options(path: Path, name: str) -> dict:
    if name not in BACKEND_KEYS:
        raise ValueError(f"Unsupported backend: {name}")
    backend_path = path.parent / "backends" / f"{name}.yaml"
    options = _mapping(backend_path)
    _keys(options, BACKEND_KEYS[name], name)
    if name in ('anygrasp', 'anygrasp_filtered', 'obb_filtered'):
        from rebot_grasp.perception.preprocessing import validate_preprocessing
        validate_preprocessing(**{k: options[k] for k in
            ('outlier_filter', 'workspace_camera', 'gripper_exclusion', 'pointcloud_quality') if k in options})
    for key in ("sdk_root", "checkpoint"):
        if key in options:
            options[key] = str((backend_path.parent / Path(options[key]).expanduser()).resolve())
    return options


def _yolo_options(raw: dict, path: Path) -> dict:
    yolo = raw.get("yolo", {})
    _keys(yolo, {"model", "device", "confidence", "iou", "classes", "model_name", "custom_classes", "use_world"}, "yolo")
    options = {key: yolo[key] for key in ("device", "confidence", "iou", "classes") if key in yolo}
    if "model" in yolo:
        options["model"] = str((path.parent / Path(yolo["model"]).expanduser()).resolve())
    elif "model_name" in yolo:
        weights = Path(yolo["model_name"]).expanduser()
        root = path.parent.parent
        if len(weights.parts) == 1:
            weights = root / "models" / weights
        options["model"] = str((root / weights).resolve())
    detection = raw.get("detection", {})
    options.setdefault("confidence", detection.get("conf_threshold", 0.5))
    options.setdefault("iou", detection.get("iou_threshold", 0.45))
    if yolo.get("use_world", True):
        options.setdefault("classes", yolo.get("custom_classes", []))
    return options


def load_config(path: str | Path, backend: str | None = None) -> AppConfig:
    path = Path(path).expanduser().resolve()
    raw = load_profile(path)
    _keys(raw, {
        "backend", "target_class", "full_scene", "filters", "planning", "yolo",
        "camera", "robot", "calibration", "detection", "grasp_pipeline", "graspnet",
    }, "application")
    name = backend or raw.get("backend", "obb")
    options = load_backend_options(path, name)
    filters = {"max_opening_m": 0.09, "opening_margin_m": 0.005, "min_score": 0.0}
    _keys(raw.get("filters", {}), set(filters), "filters")
    filters.update(raw.get("filters", {}))
    if not np.isfinite(list(filters.values())).all() or not 0 <= filters["opening_margin_m"] < filters["max_opening_m"]:
        raise ValueError("Invalid gripper filtering limits")
    planning = {"pregrasp_offset_m": 0.08, "retreat_offset_m": 0.1,
                "workspace_min_m": [-0.6, -0.6, 0.0], "workspace_max_m": [0.6, 0.6, 0.8], "T_grasp_tcp": None}
    _keys(raw.get("planning", {}), set(planning), "planning")
    planning.update(raw.get("planning", {}))
    for key in ("pregrasp_offset_m", "retreat_offset_m"):
        if not np.isfinite(planning[key]) or planning[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    lo, hi = np.asarray(planning["workspace_min_m"]), np.asarray(planning["workspace_max_m"])
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (lo < hi).all():
        raise ValueError("Invalid base-frame workspace bounds")
    if planning["T_grasp_tcp"] is not None:
        planning["T_grasp_tcp"] = checked_transform(planning["T_grasp_tcp"], "T_grasp_tcp")
    full_scene = raw.get("full_scene", False)
    label = raw.get("target_class")
    if not isinstance(full_scene, bool) or (label is not None and (not isinstance(label, str) or not label.strip())):
        raise ValueError("full_scene must be boolean; target_class must be null or a nonempty string")
    if full_scene and label is not None:
        raise ValueError("full_scene cannot be combined with target_class")
    if full_scene and name in ("obb", "obb_filtered", "anygrasp_filtered"):
        raise ValueError(f"{name} requires a target and does not support full_scene")
    hardware = {key: raw.get(key, {}) for key in ("camera", "robot", "calibration")}
    for key, value in hardware.items():
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a mapping")
    return AppConfig(name, options, label, full_scene, filters, planning, _yolo_options(raw, path), **hardware)


def load_example_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    cfg = load_profile(config_path)
    backend_dir = config_path.parent / "backends"
    if (backend_dir / "obb.yaml").is_file():
        options = load_backend_options(config_path, "obb")
        grasp = cfg.setdefault("grasp_pipeline", {}).setdefault("grasp", {})
        for key in ("depth_quantile", "insertion_depth_m"):
            if key in options:
                grasp.setdefault(key, options[key])
        if "pregrasp_offset_m" in cfg.get("planning", {}):
            grasp.setdefault("pregrasp_offset_m", cfg["planning"]["pregrasp_offset_m"])
    if (backend_dir / "graspnet.yaml").is_file():
        options = load_backend_options(config_path, "graspnet")
        options["min_depth"] = options.pop("min_depth_m", 0.05)
        options["max_depth"] = options.pop("max_depth_m", 1.0)
        options["target_class"] = cfg.get("target_class")
        cfg["graspnet"] = {**options, **cfg.get("graspnet", {})}
    return cfg
