"""Expose the tested Gemini 336 stream through the B601 camera interface."""

import numpy as np

from rebot_grasp.cameras.orbbec import OrbbecCamera

from .base import CameraDriver


class OrbbecGemini336(CameraDriver):
    def __init__(self, width: int, height: int, fps: int):
        self._camera = OrbbecCamera(width, height, fps)
        self._last_frame = None

    def open(self) -> None:
        self._camera.open()
        self._reset_frame_failures()

    def close(self) -> None:
        self._camera.close()
        self._last_frame = None

    def get_frame(self):
        try:
            frame = self._camera.capture()
        except (RuntimeError, ValueError) as exc:
            self._record_frame_failure(str(exc))
            return None, None
        self._last_frame = frame
        self._reset_frame_failures()
        depth_mm = np.clip(np.rint(frame.depth_m * 1000), 0, 65535).astype(np.uint16)
        return frame.color_bgr.copy(), depth_mm

    @property
    def K(self) -> np.ndarray:
        if self._last_frame is None:
            raise RuntimeError("Read an RGB-D frame before requesting intrinsics")
        return self._last_frame.K

    @property
    def D(self) -> np.ndarray:
        if self._last_frame is None:
            raise RuntimeError("Read an RGB-D frame before requesting distortion")
        return self._camera.color_distortion.reshape(1, -1)
