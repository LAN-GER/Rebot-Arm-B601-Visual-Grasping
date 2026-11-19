import numpy as np

from rebot_grasp.core.types import checked_transform


def transform(rotation=None, translation=None) -> np.ndarray:
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return checked_transform(result)


def inverse(T_a_b: np.ndarray) -> np.ndarray:
    T = checked_transform(T_a_b)
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def backproject(uv, depth_m: float, K: np.ndarray) -> np.ndarray:
    u, v = uv
    return np.array([(u - K[0, 2]) * depth_m / K[0, 0], (v - K[1, 2]) * depth_m / K[1, 1], depth_m])


def pose6d(T: np.ndarray) -> tuple[float, ...]:
    T = checked_transform(T)
    R = T[:3, :3]
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        rpy = (np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0]))
    else:
        rpy = (np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0)
    return tuple(float(x) for x in (*T[:3, 3], *rpy))


def hand_eye_compensation_matrix(cfg: dict) -> np.ndarray:
    calibration = cfg.get("calibration") or {}
    compensation = calibration.get("hand_eye_compensation_m") or {}
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [
        float(compensation.get("x", 0.0)),
        float(compensation.get("y", 0.0)),
        float(compensation.get("z", 0.0)),
    ]
    return T


def compose_cam_to_base_transform(T_tcp2base: np.ndarray, T_hand_eye: np.ndarray, cfg: dict) -> np.ndarray:
    T_compensation = hand_eye_compensation_matrix(cfg)
    return T_compensation @ np.asarray(T_tcp2base, dtype=np.float64) @ np.asarray(T_hand_eye, dtype=np.float64)
