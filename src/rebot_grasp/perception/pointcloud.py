from dataclasses import dataclass

import numpy as np

from rebot_grasp.core.types import RGBDFrame, Target


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray
    colors_rgb: np.ndarray
    target_mask: np.ndarray | None


def validate_outliers(*, nb_neighbors=20, std_ratio=2.0):
    if isinstance(nb_neighbors, bool) or not isinstance(nb_neighbors, int) or nb_neighbors < 2:
        raise ValueError("nb_neighbors must be an integer >= 2")
    if isinstance(std_ratio, bool) or not np.isfinite(std_ratio) or std_ratio <= 0:
        raise ValueError("std_ratio must be finite and positive")


def filter_outliers(cloud: PointCloud, *, nb_neighbors=20, std_ratio=2.0) -> PointCloud:
    """Statistical neighbor-distance filtering with aligned colors and masks."""
    validate_outliers(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(cloud.points) <= nb_neighbors:
        return cloud
    import open3d as o3d

    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(cloud.points)
    _, indices = geometry.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    return PointCloud(cloud.points[indices], cloud.colors_rgb[indices],
                      None if cloud.target_mask is None else cloud.target_mask[indices])


def from_rgbd(frame: RGBDFrame, target: Target | None, min_depth_m: float, max_depth_m: float) -> PointCloud:
    if not 0 <= min_depth_m < max_depth_m or not np.isfinite(max_depth_m):
        raise ValueError("Expected finite 0 <= min_depth_m < max_depth_m")
    if target is not None and target.mask.shape != frame.depth_m.shape:
        raise ValueError("Target mask is not aligned with depth")
    z = frame.depth_m
    valid = np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
    v, u = np.indices(z.shape)
    x = (u[valid] - frame.K[0, 2]) * z[valid] / frame.K[0, 0]
    y = (v[valid] - frame.K[1, 2]) * z[valid] / frame.K[1, 1]
    points = np.column_stack((x, y, z[valid])).astype(np.float32)
    colors = frame.color_bgr[valid][:, ::-1].astype(np.float32) / 255.0
    mask = None if target is None else target.mask[valid].copy()
    return PointCloud(points, colors, mask)


def target_pixel_mask(frame, cloud):
    """Project a filtered target cloud back into its source image."""
    eligible = np.zeros(frame.depth_m.shape, bool)
    points = cloud.points[cloud.target_mask]
    if len(points):
        u = np.rint(points[:, 0] * frame.K[0, 0] / points[:, 2] + frame.K[0, 2]).astype(int)
        v = np.rint(points[:, 1] * frame.K[1, 1] / points[:, 2] + frame.K[1, 2]).astype(int)
        eligible[v, u] = True
    return eligible
