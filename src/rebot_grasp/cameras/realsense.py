"""RealSense adapter. Construction is passive; open() is explicit."""

import time
import uuid

import numpy as np

from rebot_grasp.core.types import RGBDFrame


class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30, serial=None):
        self.width, self.height, self.fps, self.serial = width, height, fps, serial
        self._pipeline = None

    def open(self):
        if self._pipeline is not None:
            raise RuntimeError("Camera is already open")
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            profile = pipeline.start(config)
            self._align = rs.align(rs.stream.color)
            self._scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        except Exception:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            raise
        self._pipeline = pipeline

    def capture(self):
        if self._pipeline is None:
            raise RuntimeError("Call open() before capture()")
        frames = self._align.process(self._pipeline.wait_for_frames(3000))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("Missing aligned color/depth frame")
        intr = color.profile.as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        return RGBDFrame(np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()).astype(np.float32) * self._scale_m,
                         K, uuid.uuid4().hex, time.monotonic())

    def close(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
