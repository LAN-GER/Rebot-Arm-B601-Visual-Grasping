"""Shared RGB-D preprocessing for inspection and AnyGrasp inference."""
from dataclasses import dataclass

import numpy as np

from rebot_grasp.perception.pointcloud import PointCloud, from_rgbd, filter_outliers, validate_outliers
from rebot_grasp.perception.spatial_filters import (
    subset, inside_workspace, gripper_pixels, validate_workspace, validate_gripper,
)
from rebot_grasp.perception.cloud_quality import quality_settings, evaluate_quality


@dataclass(frozen=True)
class PreparedCloud:
    raw: PointCloud
    cloud: PointCloud
    report: dict
    gripper_mask: np.ndarray
    workspace_mask: np.ndarray


def validate_preprocessing(*, outlier_filter=None, workspace_camera=None,
                           gripper_exclusion=None, pointcloud_quality=None):
    if outlier_filter is not None:
        if not isinstance(outlier_filter, dict) or set(outlier_filter) - {'nb_neighbors', 'std_ratio'}:
            raise ValueError('Invalid outlier_filter settings')
        validate_outliers(**outlier_filter)
    validate_workspace(workspace_camera)
    validate_gripper(gripper_exclusion)
    quality_settings(pointcloud_quality)


def prepare_cloud(frame, target, min_depth_m=.05, max_depth_m=1., *, outlier_filter=None,
                  workspace_camera=None, gripper_exclusion=None, pointcloud_quality=None,
                  target_required=False):
    validate_preprocessing(outlier_filter=outlier_filter, workspace_camera=workspace_camera,
                           gripper_exclusion=gripper_exclusion, pointcloud_quality=pointcloud_quality)
    raw = from_rgbd(frame, target, min_depth_m, max_depth_m)
    valid = np.isfinite(frame.depth_m) & (frame.depth_m > min_depth_m) & (frame.depth_m < max_depth_m)
    target_pixels = int(target.mask.sum()) if target is not None else frame.depth_m.size
    target_bits = raw.target_mask if target is not None else np.ones(len(raw.points), bool)
    if target_required and target is None:
        target_pixels = 0
        target_bits = np.zeros(len(raw.points), bool)
    raw_target_count = int(target_bits.sum())
    gripper = gripper_pixels(frame, gripper_exclusion)
    self_bits = gripper[valid]
    region = inside_workspace(raw.points, workspace_camera)
    workspace_mask = np.zeros(frame.depth_m.shape, bool)
    workspace_mask[valid] = region
    # Keep outside-workspace scene points as collision context, steer grasps inside.
    steering = target_bits & region
    steering = steering if target_required or target is not None or workspace_camera is not None else None
    cloud = subset(PointCloud(raw.points, raw.colors_rgb, steering), ~self_bits)
    before_count = len(cloud.points)
    before_target = int(cloud.target_mask.sum()) if cloud.target_mask is not None else before_count
    if outlier_filter is not None:
        cloud = filter_outliers(cloud, **outlier_filter)
    final_target = int(cloud.target_mask.sum()) if cloud.target_mask is not None else len(cloud.points)
    metrics = dict(scene_points=len(cloud.points), target_points=final_target,
                   target_pixels=target_pixels, valid_depth_points=raw_target_count,
                   valid_depth_ratio=raw_target_count / target_pixels if target_pixels else 0.,
                   retained_ratio=final_target / raw_target_count if raw_target_count else 0.,
                   outlier_ratio=(before_target - final_target) / before_target if before_target else 0.)
    quality = evaluate_quality(metrics, pointcloud_quality, target_required=target_required,
                               target_found=target is not None)
    report = dict(frame_id=frame.frame_id, coordinate_frame='camera_optical',
                  workspace_keeps_collision_context=True,
                  settings=dict(workspace_camera=workspace_camera, gripper_exclusion=gripper_exclusion,
                                outlier_filter=outlier_filter),
                  counts=dict(raw_points=len(raw.points), raw_target_points=raw_target_count,
                              gripper_removed_points=int(self_bits.sum()),
                              gripper_removed_target_points=int((target_bits & self_bits).sum()),
                              workspace_excluded_target_points=int((target_bits & ~self_bits & ~region).sum()),
                              before_outlier_points=before_count, before_outlier_target_points=before_target,
                              outlier_removed_points=before_count - len(cloud.points),
                              outlier_removed_target_points=before_target - final_target,
                              final_points=len(cloud.points), final_target_points=final_target),
                  quality=quality)
    return PreparedCloud(raw, cloud, report, gripper, workspace_mask)
