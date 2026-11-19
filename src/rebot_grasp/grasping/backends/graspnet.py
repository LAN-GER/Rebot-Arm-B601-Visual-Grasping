"""GraspNet-Baseline with the same RGB-D and candidate interface as AnyGrasp."""

import numpy as np

from rebot_grasp.core.types import GraspCandidate, RGBDFrame, Target
from rebot_grasp.geometry.transforms import transform
from rebot_grasp.perception.pointcloud import from_rgbd

from ._graspnet_sdk import GraspNetDetector


class GraspNetBackend:
    def __init__(
        self, *, sdk_root, checkpoint, num_point=20000, num_view=300, min_depth_m=0.05,
        max_depth_m=1.0, collision_thresh=0.01, voxel_size=0.01, top_k=50,
        target_margin_px=12, target_expand_ratio=1.35, detector=None,
    ):
        for name, value in {"num_point": num_point, "num_view": num_view, "top_k": top_k}.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite([min_depth_m, max_depth_m, collision_thresh, voxel_size, target_expand_ratio]).all():
            raise ValueError("GraspNet settings must be finite")
        if not 0 <= min_depth_m < max_depth_m or collision_thresh < 0 or voxel_size <= 0:
            raise ValueError("Invalid GraspNet depth or collision settings")
        if type(target_margin_px) is not int or target_margin_px < 0 or target_expand_ratio < 1:
            raise ValueError("Invalid GraspNet target bounds")
        self._detector = detector if detector is not None else GraspNetDetector(sdk_root, checkpoint, num_view)
        self._rng = np.random.default_rng()
        self.num_point = num_point
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.top_k = top_k
        self.target_margin_px = target_margin_px
        self.target_expand_ratio = target_expand_ratio

    def predict(self, frame: RGBDFrame, target: Target | None) -> list[GraspCandidate]:
        cloud = from_rgbd(frame, target, self.min_depth_m, self.max_depth_m)
        target_points = cloud.points if cloud.target_mask is None else cloud.points[cloud.target_mask]
        if not len(target_points):
            return []
        indices = self._rng.choice(len(target_points), self.num_point, replace=len(target_points) < self.num_point)
        grasps = self._detector.predict(
            target_points[indices], cloud.points, self.collision_thresh, self.voxel_size,
        )
        bounds = self._target_bounds(target.mask) if target is not None else None
        candidates = []
        for grasp in grasps:
            center = np.asarray(grasp.translation)
            rotation = np.asarray(grasp.rotation_matrix)
            if not np.isfinite(grasp.depth) or grasp.depth < 0:
                continue
            if center.shape != (3,) or not np.isfinite(center).all() or center[2] <= 0 or rotation.shape != (3, 3):
                continue
            if bounds is not None:
                pixel = (frame.K @ center)[:2] / center[2]
                lo, hi = bounds
                if not np.all((lo <= pixel) & (pixel <= hi)):
                    continue
            try:
                tip = center + grasp.depth * rotation[:, 0]
                candidate = GraspCandidate(
                    transform(rotation, tip), float(grasp.width), float(grasp.score), "graspnet", frame.frame_id,
                )
            except ValueError:
                continue
            candidates.append(candidate)
            if len(candidates) == self.top_k:
                break
        return candidates

    def _target_bounds(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.nonzero(mask)
        lo = np.array([cols.min(), rows.min()])
        hi = np.array([cols.max(), rows.max()])
        center = (hi + lo) / 2
        radius = (hi - lo) * self.target_expand_ratio / 2 + self.target_margin_px
        return center - radius, center + radius
