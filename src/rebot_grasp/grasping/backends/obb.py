"""Mask short-axis grasping, following the upstream OBB approach.

Unlike upstream, this module accepts project Target objects, not YOLO Results.
"""

import numpy as np

from rebot_grasp.core.types import GraspCandidate, RGBDFrame, Target
from rebot_grasp.geometry.transforms import backproject, transform


class OBBBackend:
    def __init__(self, depth_quantile=0.5, insertion_depth_m=0.015, min_depth_m=0.05, max_depth_m=1.0):
        if not np.isfinite([depth_quantile, insertion_depth_m, min_depth_m, max_depth_m]).all():
            raise ValueError("OBB parameters must be finite")
        if not 0 <= depth_quantile <= 1 or insertion_depth_m < 0 or not 0 <= min_depth_m < max_depth_m:
            raise ValueError("Invalid OBB depth settings")
        self.depth_quantile = depth_quantile
        self.insertion_depth_m = insertion_depth_m
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def predict(self, frame: RGBDFrame, target: Target | None) -> list[GraspCandidate]:
        if target is None:
            return []  # OBB requires a selected object; no scene-wide fallback.
        if target.mask.shape != frame.depth_m.shape:
            raise ValueError("Target mask is not aligned with depth")
        if np.count_nonzero(target.mask) < 4:
            return []
        import cv2

        contours, _ = cv2.findContours(target.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        component = np.zeros(target.mask.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 1, cv2.FILLED)
        component = component.astype(bool) & target.mask
        corners = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
        edges = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(edges, axis=1)
        if lengths.min() < 1:
            return []
        short = edges[np.argmin(lengths)] / lengths.min()
        center = corners.mean(axis=0)
        span = float(lengths.min())

        # Refine width using the object's central cross-section, as upstream does.
        ys, xs = np.nonzero(component)
        pixels = np.column_stack((xs, ys)).astype(float)
        long_axis = np.array([-short[1], short[0]])
        along = (pixels - center) @ long_axis
        band = pixels[np.abs(along) <= max(2.0, float(lengths.max()) * 0.08)]
        if len(band) >= 8:
            offsets = (band - center) @ short
            lo, hi = np.quantile(offsets, [0.05, 0.95])
            center += short * (lo + hi) * 0.5
            span = float(hi - lo)
        if span <= 0:
            return []

        depths = frame.depth_m[component]
        depths = depths[np.isfinite(depths) & (depths > self.min_depth_m) & (depths < self.max_depth_m)]
        if not len(depths):
            return []
        z = float(np.quantile(depths, self.depth_quantile))
        center_pixel = np.rint(center).astype(int)
        u, v = center_pixel
        if not (0 <= v < component.shape[0] and 0 <= u < component.shape[1] and component[v, u]):
            return []  # Reject centers in holes or outside a concave target.
        position = backproject(center, z, frame.K)
        approach = position / np.linalg.norm(position)
        short_3d = np.array([short[0] * z / frame.K[0, 0], short[1] * z / frame.K[1, 1], 0.0])
        opening = short_3d - np.dot(short_3d, approach) * approach
        norm = np.linalg.norm(opening)
        if norm < 1e-9:
            return []
        opening /= norm
        if opening[0] < 0:
            opening = -opening
        rotation = np.column_stack((approach, opening, np.cross(approach, opening)))
        # Convert surface estimate to the shared fingertip-plane origin once.
        position = position + self.insertion_depth_m * approach
        width = float(np.linalg.norm(short_3d) * span)
        return [GraspCandidate(transform(rotation, position), width, target.confidence, "obb", frame.frame_id)]
