import numpy as np

from rebot_grasp.core.interfaces import Camera, Robot
from rebot_grasp.core.types import Capture, checked_transform


def capture_rgbd(camera: Camera, warmup_frames: int = 5) -> Capture:
    if warmup_frames < 0:
        raise ValueError("Warmup frame count must be nonnegative")
    try:
        camera.open()
        for _ in range(warmup_frames):
            camera.capture()
        return Capture(camera.capture())
    finally:
        camera.close()


def capture_stationary(camera: Camera, robot: Robot, T_tcp_camera: np.ndarray,
                       position_tolerance_m=0.001, rotation_tolerance_rad=0.01) -> Capture:
    """Bracket capture with pose reads; caller must hold the arm stationary.

    This rejects observed motion; it is not hardware timestamp synchronization.
    """
    if min(position_tolerance_m, rotation_tolerance_rad) < 0 or not np.isfinite([position_tolerance_m, rotation_tolerance_rad]).all():
        raise ValueError("Pose tolerances must be finite and nonnegative")
    before = checked_transform(robot.tcp_pose())
    frame = camera.capture()
    after = checked_transform(robot.tcp_pose())
    translation = np.linalg.norm(before[:3, 3] - after[:3, 3])
    angle = np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1, 1))
    if translation > position_tolerance_m or angle > rotation_tolerance_rad:
        raise RuntimeError("Robot moved during RGB-D capture; discard frame")
    return Capture(frame, before @ checked_transform(T_tcp_camera))
