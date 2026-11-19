"""Translate Ultralytics results into SDK-independent Target objects."""

from pathlib import Path

import numpy as np

from rebot_grasp.core.types import RGBDFrame, Target


def _array(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


class YoloDetector:
    def __init__(self, model, device="cpu", confidence=0.5, iou=0.45, classes=None):
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError("YOLO confidence and iou must be in [0,1]")
        self.path = Path(model)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.classes = classes or []
        self._model = None

    def start(self):
        if self._model is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"YOLO checkpoint not found: {self.path}; install it explicitly")
            from ultralytics import YOLO
            self._model = YOLO(str(self.path))
            if self.classes and any(name in self.path.name.lower() for name in ("yoloe", "world")):
                self._model.set_classes(self.classes)

    def detect(self, frame: RGBDFrame) -> list[Target]:
        import cv2
        self.start()
        targets = []
        for result in self._model.predict(frame.color_bgr, device=self.device, conf=self.confidence, iou=self.iou, verbose=False):
            obb = result.obb
            boxes = obb if obb is not None and len(obb) else result.boxes
            if boxes is None:
                continue
            for index in range(len(boxes)):
                mask = np.zeros(frame.depth_m.shape, dtype=np.uint8)
                if boxes is obb:
                    polygon = _array(obb.xyxyxyxy[index])
                    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
                elif result.masks is not None:
                    # Original-image polygons avoid resizing a letterboxed tensor.
                    polygon = np.asarray(result.masks.xy[index])
                    if len(polygon) >= 3:
                        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
                else:
                    x1, y1, x2, y2 = np.rint(_array(boxes.xyxy[index])).astype(int)
                    h, w = mask.shape
                    x1, x2 = np.clip([x1, x2], 0, w)
                    y1, y2 = np.clip([y1, y2], 0, h)
                    mask[y1:y2, x1:x2] = 1
                cls_id = int(_array(boxes.cls[index]).item())
                targets.append(Target(str(result.names[cls_id]), mask, float(_array(boxes.conf[index]).item())))
        return targets
