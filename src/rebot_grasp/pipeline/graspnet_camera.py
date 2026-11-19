"""GraspNet helpers that sit above camera frames and YOLO detections."""

from __future__ import annotations

from pathlib import Path
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np


from rebot_grasp.grasping.backends import _graspnet_sdk

from rebot_grasp.paths import PROJECT_ROOT
from rebot_grasp.geometry.grasp_frames import graspnet_rotation_to_rebot_tcp_rotation, transform_grasp_pose_to_base_with_retreat
from rebot_grasp.perception.detections import YoloDetection, detect_objects
if TYPE_CHECKING:
    import open3d as o3d
    from graspnet import GraspNet
    from graspnetAPI import Grasp, GraspGroup

GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
DEFAULT_NUM_VIEW = 300
DEFAULT_VOXEL_SIZE = 0.01
DEFAULT_WARMUP_FRAMES = 20


@dataclass
class GraspNetFrameResult:
    grasps: GraspGroup
    pre_bbox_grasps: GraspGroup
    bbox_grasps: GraspGroup
    best: Optional[Grasp]
    status: str
    target_status: str
    detections: list[YoloDetection]
    selected_target: Optional[YoloDetection]
    o3d_cloud: o3d.geometry.PointCloud
    raw_cloud: np.ndarray


def copy_grasp_group(grasps: GraspGroup) -> GraspGroup:
    return type(grasps)(grasps.grasp_group_array.copy())


def resolve_checkpoint_path(
    checkpoint: str,
    *,
    project_root: Path = PROJECT_ROOT,
    graspnet_root: Path = GRASPNET_ROOT,
) -> Path:
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if len(checkpoint_path.parts) > 1:
        return project_root / checkpoint_path
    return graspnet_root / "checkpoints" / checkpoint_path


def build_net(checkpoint_path: str | Path, num_view: int = DEFAULT_NUM_VIEW, *, sdk_root: str | Path = GRASPNET_ROOT):
    sdk_root = Path(sdk_root)
    checkpoint = resolve_checkpoint_path(str(checkpoint_path), graspnet_root=sdk_root)
    _graspnet_sdk.prepare_imports(sdk_root)
    return _graspnet_sdk.build_net(checkpoint, num_view)


def _expanded_bbox_pixel_bounds(
    bbox_xyxy: tuple[int, int, int, int],
    expand_ratio: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Pixel bounds of a bbox expanded about its center, clipped to the image."""
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    x1 = int(np.clip(int(round(cx - half_w)), 0, max(0, width - 1)))
    y1 = int(np.clip(int(round(cy - half_h)), 0, max(0, height - 1)))
    x2 = int(np.clip(int(round(cx + half_w)), 0, max(0, width - 1)))
    y2 = int(np.clip(int(round(cy + half_h)), 0, max(0, height - 1)))
    return x1, y1, x2, y2


def build_end_points(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    num_point: int,
    min_depth_m: float,
    max_depth_m: float,
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None,
    expand_ratio: float = 1.0,
) -> tuple[dict, o3d.geometry.PointCloud, np.ndarray]:
    import cv2
    import open3d as o3d
    import torch
    from data_utils import CameraInfo, create_point_cloud_from_depth_image

    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(depth_mm, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    min_mm = int(max(0.0, min_depth_m) * 1000.0)
    max_mm = int(max_depth_m * 1000.0)
    depth = depth_mm.astype(np.uint16, copy=False)
    mask = (depth > min_mm) & (depth < max_mm)
    if int(mask.sum()) == 0:
        raise RuntimeError("No valid depth pixels in the configured depth range.")

    # Concentrate network input points on the target bbox when available;
    # collision detection / visualization still use the full masked scene cloud.
    sample_mask = mask
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = _expanded_bbox_pixel_bounds(bbox_xyxy, expand_ratio, depth.shape[1], depth.shape[0])
        bbox_mask = np.zeros_like(mask)
        bbox_mask[y1 : y2 + 1, x1 : x2 + 1] = True
        cropped = mask & bbox_mask
        if int(cropped.sum()) >= 16:
            sample_mask = cropped

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = depth.shape
    camera = CameraInfo(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), 1000.0)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_masked = cloud[sample_mask]
    color_masked = color_rgb[sample_mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs = np.concatenate(
            [
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True),
            ],
            axis=0,
        )

    end_points = {
        "point_clouds": torch.from_numpy(cloud_masked[idxs].astype(np.float32)[np.newaxis]).cuda(non_blocking=True),
        "cloud_colors": color_masked[idxs],
    }

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud[mask].astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_rgb[mask].astype(np.float32))
    return end_points, o3d_cloud, cloud[mask]


def infer_grasps(
    net: GraspNet,
    end_points: dict,
    raw_cloud: np.ndarray,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> tuple[GraspGroup, dict[str, int]]:
    gg, decoded_count = _graspnet_sdk.infer_group(net, end_points, raw_cloud, collision_thresh, voxel_size)

    return gg, {
        "decoded": decoded_count,
        "pre_collision": decoded_count,
        "collision_removed": decoded_count - len(gg),
        "final": len(gg),
    }


def filter_grasps_by_bbox(
    grasps: GraspGroup,
    bbox_xyxy: tuple[int, int, int, int],
    K: np.ndarray,
    *,
    margin_px: int = 0,
    expand_ratio: float = 1.0,
    image_shape: Optional[tuple[int, int]] = None,
) -> GraspGroup:
    if len(grasps) == 0:
        return grasps

    translations = np.asarray(grasps.translations, dtype=np.float64)
    z = translations[:, 2]
    valid_z = z > 1e-6
    u = np.full(len(grasps), np.nan, dtype=np.float64)
    v = np.full(len(grasps), np.nan, dtype=np.float64)
    u[valid_z] = float(K[0, 0]) * translations[valid_z, 0] / z[valid_z] + float(K[0, 2])
    v[valid_z] = float(K[1, 1]) * translations[valid_z, 1] / z[valid_z] + float(K[1, 2])

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    x1 = int(round(cx - half_w)) - int(margin_px)
    y1 = int(round(cy - half_h)) - int(margin_px)
    x2 = int(round(cx + half_w)) + int(margin_px)
    y2 = int(round(cy + half_h)) + int(margin_px)
    if image_shape is not None:
        h, w = image_shape
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        x2 = int(np.clip(x2, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        y2 = int(np.clip(y2, 0, max(0, h - 1)))

    keep = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return grasps[keep]


def filter_grasps_by_width(grasps: GraspGroup, max_width_m: Optional[float]) -> GraspGroup:
    if max_width_m is None or len(grasps) == 0:
        return grasps
    return grasps[np.asarray(grasps.widths, dtype=np.float64) <= float(max_width_m)]


def select_best_grasp(grasps: GraspGroup) -> Optional[Grasp]:
    if len(grasps) == 0:
        return None
    ranked = copy_grasp_group(grasps)
    try:
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS skipped: {exc}")
    ranked.sort_by_score()
    return ranked[0] if len(ranked) > 0 else None


def select_target(detections: list[YoloDetection], target_class: Optional[str]) -> Optional[YoloDetection]:
    if not detections:
        return None
    candidates = detections
    if target_class:
        target_norm = target_class.casefold()
        exact = [target for target in detections if target.class_name.casefold() == target_norm]
        contains = [target for target in detections if target_norm in target.class_name.casefold()]
        candidates = exact or contains
    if not candidates:
        return None
    return max(candidates, key=lambda target: target.conf)


def selected_target_text(selected: Optional[YoloDetection], target_class: Optional[str]) -> str:
    if selected is None:
        return f"target={target_class or 'best'} not found"
    return f"target={selected.class_name} {selected.conf:.2f}"


def target_status_text(selected: Optional[YoloDetection], detections: list[YoloDetection], target_class: Optional[str]) -> str:
    if selected is not None:
        return f"target={selected.class_name} {selected.conf:.2f} detections={len(detections)}"
    if target_class:
        return f"target={target_class} not found detections={len(detections)}"
    return f"target not found detections={len(detections)}"


def grasp_to_base_poses(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return transform_grasp_pose_to_base_with_retreat(
        np.asarray(grasp.translation, dtype=np.float64),
        graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix),
        T_cam2base,
        pregrasp_offset_m,
        retreat_offset_m,
        insertion_depth_m,
    )


def infer_frame(
    net: GraspNet,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    num_point: int,
    min_depth: float,
    max_depth: float,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    yolo_model: Optional[Any] = None,
    yolo_opts: Optional[dict[str, Any]] = None,
    target_class: Optional[str] = None,
    target_margin_px: int = 0,
    target_expand_ratio: float = 1.0,
    max_grasp_width_m: Optional[float] = None,
) -> GraspNetFrameResult:
    import open3d as o3d
    from graspnetAPI import GraspGroup

    detections: list[YoloDetection] = []
    selected_target: Optional[YoloDetection] = None
    target_label = "full scene"

    if yolo_model is not None:
        _, detections = detect_objects(yolo_model, color_bgr, yolo_opts or {})
        selected_target = select_target(detections, target_class)
        if selected_target is None:
            target_status = target_status_text(selected_target, detections, target_class)
            empty = GraspGroup()
            empty_cloud = o3d.geometry.PointCloud()
            return GraspNetFrameResult(
                grasps=empty,
                pre_bbox_grasps=empty,
                bbox_grasps=empty,
                best=None,
                status=f"inference skipped: {target_status}",
                target_status=target_status,
                detections=detections,
                selected_target=None,
                o3d_cloud=empty_cloud,
                raw_cloud=np.empty((0, 3), dtype=np.float32),
            )
        target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

    tic = time.time()
    crop_bbox = selected_target.bbox_xyxy if selected_target is not None else None
    end_points, o3d_cloud, raw_cloud = build_end_points(
        color_bgr,
        depth_mm,
        K,
        num_point,
        min_depth,
        max_depth,
        bbox_xyxy=crop_bbox,
        expand_ratio=target_expand_ratio,
    )
    grasps, counts = infer_grasps(net, end_points, raw_cloud, collision_thresh, voxel_size)
    pre_bbox_grasps = copy_grasp_group(grasps)
    if selected_target is not None:
        grasps = filter_grasps_by_bbox(
            grasps,
            selected_target.bbox_xyxy,
            K,
            margin_px=target_margin_px,
            expand_ratio=target_expand_ratio,
            image_shape=color_bgr.shape[:2],
        )
    bbox_grasps = copy_grasp_group(grasps)
    grasps = filter_grasps_by_width(grasps, max_grasp_width_m)
    best = select_best_grasp(grasps)
    elapsed = time.time() - tic

    if yolo_model is None:
        status = f"grasps={len(grasps)} decoded={counts['decoded']} inference={elapsed:.2f}s"
        target_status = "YOLO disabled: full-scene GraspNet"
    else:
        status = (
            f"{target_label} grasps={len(grasps)}/{len(bbox_grasps)}/{len(pre_bbox_grasps)} decoded={counts['decoded']} "
            f"collide={counts['collision_removed']}/{counts['pre_collision']} inference={elapsed:.2f}s"
        )
        target_status = target_status_text(selected_target, detections, target_class)

    return GraspNetFrameResult(
        grasps=grasps,
        pre_bbox_grasps=pre_bbox_grasps,
        bbox_grasps=bbox_grasps,
        best=best,
        status=status,
        target_status=target_status,
        detections=detections,
        selected_target=selected_target,
        o3d_cloud=o3d_cloud,
        raw_cloud=raw_cloud,
    )
