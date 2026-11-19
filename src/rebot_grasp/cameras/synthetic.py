"""Explicit synthetic input for software demonstration, never a hardware fallback."""

import time

import numpy as np

from rebot_grasp.core.types import Capture, RGBDFrame, Target
from rebot_grasp.geometry.transforms import transform


def demo_capture() -> tuple[Capture, Target]:
    color = np.full((120, 160, 3), 65, dtype=np.uint8)
    depth = np.full((120, 160), 0.60, dtype=np.float32)
    mask = np.zeros((120, 160), dtype=bool)
    mask[36:84, 72:88] = True
    color[mask] = [40, 180, 220]
    depth[mask] = 0.45
    K = np.array([[180, 0, 80], [0, 180, 60], [0, 0, 1]], dtype=float)
    frame = RGBDFrame(color, depth, K, "synthetic-demo", time.monotonic())
    T_base_camera = transform(np.diag([1.0, -1.0, -1.0]), [0.30, 0.0, 0.80])
    return Capture(frame, T_base_camera, synthetic=True), Target("synthetic-block", mask)
