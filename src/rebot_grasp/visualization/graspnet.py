"""Image overlays and an interactive GraspNet candidate viewer."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from rebot_grasp.perception.detections import YoloDetection
from rebot_grasp.pipeline.graspnet_camera import GraspNetFrameResult, copy_grasp_group

if TYPE_CHECKING:
    import open3d as o3d
    from graspnetAPI import Grasp, GraspGroup

DISPLAY_FLIP_X = np.diag([1., -1., -1., 1.])


class Open3DGraspWindow:
    def __init__(self, title: str, top_k: int) -> None:
        import open3d as o3d
        self._top_k = top_k
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(title, width=1280, height=720):
            self._vis.destroy_window()
            raise RuntimeError("Open3D visualizer window could not be created")
        self._geometries = []
        self._initialized = False

    def update(self, cloud: o3d.geometry.PointCloud, grasps: GraspGroup) -> None:
        import open3d as o3d
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []

        cloud_vis = o3d.geometry.PointCloud(cloud)
        cloud_vis.transform(DISPLAY_FLIP_X)
        geometries = [cloud_vis]

        if len(grasps) > 0:
            grasps_vis = copy_grasp_group(grasps)
            try:
                grasps_vis = grasps_vis.nms()
            except Exception as exc:
                print(f"Grasp NMS skipped: {exc}")
            grasps_vis.sort_by_score()
            grasps_vis = grasps_vis[: self._top_k]
            grasps_vis.transform(DISPLAY_FLIP_X)
            geometries.extend(grasps_vis.to_open3d_geometry_list())

        for geom in geometries:
            self._vis.add_geometry(geom, reset_bounding_box=not self._initialized)
        self._geometries = geometries
        self._initialized = True
        self.poll()

    def poll(self) -> bool:
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        self._vis.destroy_window()


def visualization_grasps(result: GraspNetFrameResult, mode: str) -> GraspGroup:
    """Return the grasp set used for Open3D diagnostics."""
    if mode == "pre-bbox":
        return result.pre_bbox_grasps
    if mode == "bbox":
        return result.bbox_grasps
    return result.grasps


def draw_detections_overlay(
    frame: np.ndarray,
    detections: list[YoloDetection],
    selected: Optional[YoloDetection],
    target_class: Optional[str],
) -> np.ndarray:
    import cv2
    display = frame.copy()
    selected_key = None
    if selected is not None:
        selected_key = (selected.result_index, selected.detection_index)
    for target in detections:
        is_selected = selected_key == (target.result_index, target.detection_index)
        color = (0, 255, 80) if is_selected else (0, 185, 255)
        thickness = 3 if is_selected else 2
        x1, y1, x2, y2 = target.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        label = f"{target.class_name} {target.conf:.2f}"
        if target_class and is_selected:
            label = f"TARGET {label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(display, (x1, bg_y1), (x1 + label_size[0] + 8, y1), (0, 0, 0), -1)
        cv2.putText(display, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def draw_best_grasp_projection(display: np.ndarray, grasp: Optional[Grasp], K: np.ndarray) -> None:
    import cv2
    if grasp is None:
        return
    x, y, z = [float(v) for v in grasp.translation]
    if z <= 1e-6:
        return
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
        cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        label = f"best score={grasp.score:.2f} width={grasp.width * 100:.1f}cm"
        cv2.putText(display, label, (u + 10, max(24, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def draw_status(
    frame: np.ndarray,
    status: str,
    target_status: str = "",
    frozen: bool = False,
    title: str = "GraspNet Full-Scene Demo",
) -> np.ndarray:
    import cv2
    display = frame.copy()
    lines = [
        title,
        "G/SPACE: infer   R: resume   Q/ESC: quit",
    ]
    if target_status:
        lines.append(target_status)
    lines.append(status)
    y = 28
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    if frozen:
        cv2.putText(display, "[FROZEN]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    return display
