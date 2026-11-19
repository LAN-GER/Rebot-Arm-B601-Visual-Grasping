"""Camera-frame grasp regions and fixed-view self masks, independent of SDKs."""
import numpy as np

from rebot_grasp.perception.pointcloud import PointCloud


def subset(cloud, keep):
    return PointCloud(cloud.points[keep], cloud.colors_rgb[keep],
                      None if cloud.target_mask is None else cloud.target_mask[keep])


def validate_workspace(settings):
    if settings is None:
        return
    if not isinstance(settings, dict) or set(settings) != {'min_m', 'max_m'}:
        raise ValueError('workspace_camera requires min_m and max_m')
    lo, hi = np.asarray(settings['min_m']), np.asarray(settings['max_m'])
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (lo < hi).all():
        raise ValueError('workspace_camera requires finite, ordered XYZ bounds in meters')


def inside_workspace(points, settings):
    if settings is None:
        return np.ones(len(points), bool)
    return ((points >= settings['min_m']) & (points <= settings['max_m'])).all(axis=1)


def validate_gripper(settings):
    if settings is None:
        return
    required = {'reference_resolution', 'polygons_uv', 'min_depth_m', 'max_depth_m'}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f'gripper_exclusion requires {sorted(required)}')
    size = settings['reference_resolution']
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError('reference_resolution must be [width, height] in pixels')
    lo, hi = settings['min_depth_m'], settings['max_depth_m']
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
        raise ValueError('Invalid gripper depth interval')
    polygons = settings['polygons_uv']
    if not isinstance(polygons, list) or not polygons:
        raise ValueError('gripper_exclusion needs at least one polygon')
    for polygon in polygons:
        p = np.asarray(polygon)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
            raise ValueError('Gripper polygons require >= 3 normalized [u,v] vertices in [0,1]')
        area = abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
        if area < 1e-8:
            raise ValueError('Gripper polygon has zero area')


def gripper_pixels(frame, settings):
    """Only remove the configured silhouette within its near-depth interval.

    This is a mount-specific conservative mask, not a robot CAD/self model.
    Reject different resolutions rather than silently reuse a mismatched profile.
    """
    mask = np.zeros(frame.depth_m.shape, dtype=np.uint8)
    if settings is None:
        return mask.astype(bool)
    h, w = mask.shape
    if [w, h] != settings['reference_resolution']:
        raise ValueError('Gripper profile resolution mismatch; configure the current camera view')
    import cv2
    for polygon in settings['polygons_uv']:
        vertices = np.rint(np.asarray(polygon) * [w - 1, h - 1]).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 1)
    return (mask.astype(bool) & (frame.depth_m >= settings['min_depth_m'])
            & (frame.depth_m <= settings['max_depth_m']))
