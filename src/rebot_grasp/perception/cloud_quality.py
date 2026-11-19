"""Explainable point-cloud admission checks; no camera, detector or robot calls."""
import numpy as np


DEFAULTS = dict(min_scene_points=1, min_target_points=1,
                min_valid_depth_ratio=0., min_retained_ratio=0., max_outlier_ratio=1.)


def quality_settings(settings):
    if settings is not None and (not isinstance(settings, dict) or set(settings) - set(DEFAULTS)):
        raise ValueError('Unknown pointcloud_quality settings')
    result = {**DEFAULTS, **(settings or {})}
    for key in ('min_scene_points', 'min_target_points'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be an integer >= 1')
    for key in ('min_valid_depth_ratio', 'min_retained_ratio', 'max_outlier_ratio'):
        if isinstance(result[key], bool) or not np.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    return result


def evaluate_quality(metrics, settings=None, *, target_required=False, target_found=True):
    limits = quality_settings(settings)
    reasons = []
    if target_required and not target_found:
        reasons.append('target_not_found')
    if metrics['scene_points'] < limits['min_scene_points']:
        reasons.append('insufficient_scene_points')
    # In full-scene mode target_points means the eligible grasp region.
    if metrics['target_points'] < limits['min_target_points']:
        reasons.append('insufficient_target_points')
    if metrics['valid_depth_ratio'] < limits['min_valid_depth_ratio']:
        reasons.append('insufficient_valid_depth')
    if metrics['retained_ratio'] < limits['min_retained_ratio']:
        reasons.append('excessive_region_removal')
    if metrics['outlier_ratio'] > limits['max_outlier_ratio']:
        reasons.append('excessive_outliers')
    return dict(passed=not reasons, reasons=reasons, metrics=metrics, thresholds=limits)
