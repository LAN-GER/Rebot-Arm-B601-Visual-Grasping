import numpy as np

from rebot_grasp.core.types import GraspCandidate, checked_transform


def grasp_to_tcp(candidate: GraspCandidate, T_base_camera: np.ndarray, T_grasp_tcp: np.ndarray) -> np.ndarray:
    return checked_transform(T_base_camera) @ candidate.T_camera_grasp @ checked_transform(T_grasp_tcp)


def retreat_along_approach(T_base_tcp: np.ndarray, approach_base: np.ndarray, distance_m: float) -> np.ndarray:
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("Retreat distance must be finite and nonnegative")
    axis = np.asarray(approach_base, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
        raise ValueError("Approach must be a unit vector")
    result = checked_transform(T_base_tcp).copy()
    result[:3, 3] -= axis * distance_m
    return checked_transform(result)


def canonicalize_gripper_plan(plan):
    """Use the same symmetric-jaw orientation branch as the B601 transform."""
    from dataclasses import replace
    from rebot_grasp.geometry.grasp_frames import canonicalize_parallel_gripper_tcp_rotation

    poses = {}
    for name in ("T_base_pregrasp", "T_base_grasp", "T_base_retreat"):
        pose = getattr(plan, name).copy()
        pose[:3, :3] = canonicalize_parallel_gripper_tcp_rotation(pose[:3, :3])
        poses[name] = pose
    return replace(plan, **poses)
