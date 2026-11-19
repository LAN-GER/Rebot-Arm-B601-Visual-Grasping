"""Show scene points and candidate jaws in the RGB-D camera frame."""

import numpy as np

from rebot_grasp.core.types import GraspCandidate, RGBDFrame
from rebot_grasp.perception.pointcloud import from_rgbd


def _jaw_segments(candidate: GraspCandidate) -> np.ndarray:
    pose = candidate.T_camera_grasp
    midpoint, approach, opening = pose[:3, 3], pose[:3, 0], pose[:3, 1]
    half_width = candidate.width_m * opening / 2
    back = midpoint - 0.03 * approach
    return np.array([midpoint - half_width, back - half_width, back + half_width, midpoint + half_width])


def show_pointcloud(
    frame: RGBDFrame,
    candidates: list[GraspCandidate],
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 1.0,
    top_k: int = 20,
) -> None:
    cloud = from_rgbd(frame, None, min_depth_m, max_depth_m)
    if not len(cloud.points):
        raise ValueError("No valid depth points to display")

    import open3d as o3d

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(cloud.points)
    points.colors = o3d.utility.Vector3dVector(cloud.colors_rgb)
    geometries = [points]
    for index, candidate in enumerate(candidates[:top_k]):
        jaws = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(_jaw_segments(candidate)),
            lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3]]),
        )
        jaws.paint_uniform_color([0.0, 1.0, 0.35] if index == 0 else [0.9, 0.6, 0.15])
        geometries.append(jaws)
    o3d.visualization.draw_geometries(geometries, window_name="B601 grasp preview")
