"""All public distances are meters, rotations are proper SO(3) matrices."""

from dataclasses import dataclass

import numpy as np


def checked_transform(value: np.ndarray, name: str = "transform") -> np.ndarray:
    value = np.array(value, dtype=np.float64, copy=True)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError(f"{name} must contain a proper rotation")
    value.setflags(write=False)
    return value


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    frame_id: str
    captured_at: float  # Host monotonic clock; not a hardware-synchronized timestamp.

    def __post_init__(self):
        color = _readonly(self.color_bgr)
        depth = _readonly(self.depth_m, np.float32)
        intrinsic = _readonly(self.K, np.float64)
        if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
            raise ValueError("color_bgr must be uint8 HxWx3")
        if depth.shape != color.shape[:2] or min(depth.shape) == 0:
            raise ValueError("depth_m must be nonempty and aligned with color_bgr")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("K must be a finite 3x3 camera matrix")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
            raise ValueError("K has invalid focal lengths or homogeneous row")
        if not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], 0):
            raise ValueError("Only zero-skew pinhole intrinsics are supported")
        if not self.frame_id or not np.isfinite(self.captured_at):
            raise ValueError("frame_id and finite captured_at are required")
        object.__setattr__(self, "color_bgr", color)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "K", intrinsic)


@dataclass(frozen=True)
class Target:
    label: str
    mask: np.ndarray
    confidence: float = 1.0

    def __post_init__(self):
        mask = _readonly(self.mask, bool)
        if mask.ndim != 2 or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Target requires a 2D mask and confidence in [0, 1]")
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class GraspCandidate:
    # Origin: midpoint of modeled fingertip plane. X approach, Y jaw opening.
    T_camera_grasp: np.ndarray
    width_m: float
    score: float
    backend: str
    frame_id: str

    def __post_init__(self):
        object.__setattr__(self, "T_camera_grasp", checked_transform(self.T_camera_grasp))
        if not np.isfinite([self.width_m, self.score]).all() or self.width_m <= 0:
            raise ValueError("Grasp width must be positive; width and score must be finite")
        if not self.backend or not self.frame_id:
            raise ValueError("Grasp backend and frame_id are required")


@dataclass(frozen=True)
class Capture:
    frame: RGBDFrame
    T_base_camera: np.ndarray | None = None
    synthetic: bool = False

    def __post_init__(self):
        if not isinstance(self.synthetic, bool):
            raise ValueError("synthetic must be boolean")
        if self.T_base_camera is not None:
            object.__setattr__(self, "T_base_camera", checked_transform(self.T_base_camera))


@dataclass(frozen=True)
class GraspPlan:
    candidate: GraspCandidate
    T_base_pregrasp: np.ndarray
    T_base_grasp: np.ndarray
    T_base_retreat: np.ndarray
    opening_m: float

    def __post_init__(self):
        for name in ("T_base_pregrasp", "T_base_grasp", "T_base_retreat"):
            object.__setattr__(self, name, checked_transform(getattr(self, name), name))
        if not np.isfinite(self.opening_m) or self.opening_m < self.candidate.width_m:
            raise ValueError("Planned opening must accommodate candidate width")


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    stage: str
    message: str
