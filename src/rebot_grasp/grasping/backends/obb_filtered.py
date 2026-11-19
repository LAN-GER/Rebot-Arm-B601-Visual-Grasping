"""B601 OBB on the full 2D silhouette, with filtered depth and quality checks."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from rebot_grasp.core.types import GraspCandidate
from rebot_grasp.geometry.transforms import transform
from rebot_grasp.perception.pointcloud import target_pixel_mask
from rebot_grasp.perception.preprocessing import prepare_cloud, validate_preprocessing
from rebot_grasp.perception.spatial_filters import inside_workspace


class _MaskArray:
    """Minimal CPU mask view for the preserved B601 estimator's tensor interface."""
    def __init__(self, mask):
        self.mask = mask

    def cpu(self):
        return self

    def numpy(self):
        return self.mask


def estimate_contour_pose(frame, target, depth_m, depth_quantile):
    """Adapt one aligned Target to the unchanged B601 estimate_grasp function."""
    from rebot_grasp.grasping.contour_obb import estimate_grasp, _rect_from_mask
    mask = target.mask.astype(np.uint8)
    rect = _rect_from_mask(mask)
    if rect is None:
        return None
    y, x = np.nonzero(mask)
    result = SimpleNamespace(
        names={0: target.label}, orig_shape=mask.shape,
        obb=SimpleNamespace(cls=np.array([0]), conf=np.array([target.confidence]),
                            xyxy=np.array([[x.min(), y.min(), x.max(), y.max()]]),
                            xyxyxyxy=rect[None, ...]),
        boxes=[None], masks=SimpleNamespace(data=[_MaskArray(mask)]),
    )
    # All pixels outside the filtered target are zero in the filtered branch,
    # so B601's small center-depth fallback cannot resurrect rejected points.
    depth_mm = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m * 1000., 0).astype(np.float32)
    return estimate_grasp(result, 0, depth_mm, frame.K, depth_quantile=depth_quantile)


def pose_record(pose):
    if pose is None:
        return None
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(pose).items()}


class FilteredOBBBackend:
    def __init__(self, *, depth_quantile=.5, min_depth_m=.05, max_depth_m=1., insertion_depth_m=0.,
                 outlier_filter=None, workspace_camera=None, gripper_exclusion=None,
                 pointcloud_quality=None):
        if not np.isfinite([depth_quantile, min_depth_m, max_depth_m]).all() or not 0 <= depth_quantile <= 1:
            raise ValueError('Invalid OBB quantile/depth settings')
        if not 0 <= min_depth_m < max_depth_m:
            raise ValueError('Invalid OBB depth interval')
        if isinstance(insertion_depth_m, bool) or not np.isfinite(insertion_depth_m) or insertion_depth_m < 0:
            raise ValueError('insertion_depth_m must be finite and nonnegative')
        self.insertion_depth_m = float(insertion_depth_m)
        self.depth_quantile = depth_quantile
        self.min_depth_m, self.max_depth_m = min_depth_m, max_depth_m
        self.preprocessing = dict(outlier_filter=outlier_filter, workspace_camera=workspace_camera,
                                  gripper_exclusion=gripper_exclusion, pointcloud_quality=pointcloud_quality)
        validate_preprocessing(**self.preprocessing)
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.last_preprocessing_report = None
        self.last_prepared = None
        self.last_target = None
        self.last_eligible_mask = None
        self.last_comparison = None

    def start(self):
        pass

    def close(self):
        pass

    def predict(self, frame, target):
        self.reset_diagnostics()
        if target is None:
            return []
        prepared = prepare_cloud(frame, target, self.min_depth_m, self.max_depth_m,
                                 target_required=True, **self.preprocessing)
        self.last_prepared, self.last_target = prepared, target
        self.last_preprocessing_report = prepared.report
        eligible = target_pixel_mask(frame, prepared.cloud)
        self.last_eligible_mask = eligible
        raw = estimate_contour_pose(frame, target, frame.depth_m, self.depth_quantile)
        filtered = None
        status = 'pointcloud_rejected'
        if prepared.report['quality']['passed']:
            # Preserve the full RGB silhouette for the rectangle, center and
            # short-axis refinement. Filter only the depth samples: holes and
            # rejected points must not shrink or rotate the 2D grasp geometry.
            filtered = estimate_contour_pose(frame, target,
                                     np.where(eligible, frame.depth_m, 0), self.depth_quantile)
            status = 'estimated' if filtered is not None and filtered.is_valid else 'no_valid_filtered_obb'
        comparison = dict(frame_id=frame.frame_id, target_class=target.label,
                          status=status, raw=pose_record(raw), filtered=pose_record(filtered),
                          raw_is_reference_only=True, depth_quantile=self.depth_quantile,
                          algorithm_version='obb_full_mask_filtered_depth_v2',
                          geometry_source='full_2d_segmentation', depth_source='filtered_target_points',
                          insertion_depth_m=self.insertion_depth_m, candidate_position_m=None,
                          insertion_offset_camera_m=None, delta=None)
        self.last_comparison = comparison
        if filtered is None or not filtered.is_valid:
            return []
        if raw is not None and raw.is_valid:
            angle = np.arccos(np.clip((np.trace(raw.tcp_rotation.T @ filtered.tcp_rotation) - 1) / 2, -1, 1))
            comparison['delta'] = dict(position_m=(filtered.position - raw.position).tolist(),
                                       center_px=(np.array(filtered.center_px) - raw.center_px).tolist(),
                                       jaw_width_m=float(filtered.jaw_width_m - raw.jaw_width_m),
                                       rotation_angle_rad=float(angle))
        insertion_offset = self.insertion_depth_m * filtered.tcp_rotation[:, 0]
        candidate_position = filtered.position + insertion_offset
        comparison['candidate_position_m'] = candidate_position.tolist()
        comparison['insertion_offset_camera_m'] = insertion_offset.tolist()
        if not inside_workspace(candidate_position[None, :], self.preprocessing['workspace_camera'])[0]:
            comparison['status'] = 'inserted_center_outside_workspace'
            return []
        # Apply only once along tool +X. B601 execution receives the final pose.
        return [GraspCandidate(transform(filtered.tcp_rotation, candidate_position),
                               filtered.jaw_width_m, filtered.conf, 'obb_filtered', frame.frame_id)]
