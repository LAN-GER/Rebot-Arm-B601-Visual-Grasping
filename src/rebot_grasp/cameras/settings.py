"""Apply camera stream overrides for interactive examples."""
from __future__ import annotations

import argparse
from typing import Any

from rebot_grasp.cameras.streams import CameraDriver, make_camera


def configure_camera(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> dict[str, Any]:
    cam_cfg = cfg.setdefault("camera", {})
    camera_type = getattr(args, "camera_type", None)
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    fps = getattr(args, "fps", None)

    if camera_type is not None:
        cam_cfg["type"] = camera_type
    cam_type = str(cam_cfg.get("type", "")).lower()
    if not cam_type:
        raise ValueError("camera.type is missing in config; pass --camera-type or set it in YAML")

    if width is not None:
        cam_cfg["color_width"] = int(width)
        cam_cfg["depth_width"] = int(width)
    if height is not None:
        cam_cfg["color_height"] = int(height)
        cam_cfg["depth_height"] = int(height)
    if fps is not None:
        cam_cfg["fps"] = int(fps)
    elif camera_type is not None and realsense_default_fps is not None and "realsense" in cam_type:
        cam_cfg["fps"] = int(realsense_default_fps)
    return cfg


def create_camera_from_args(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> CameraDriver:
    return make_camera(configure_camera(cfg, args, realsense_default_fps=realsense_default_fps))
