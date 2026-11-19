"""Orbbec RGB-D capture using a supported hardware depth-to-color profile."""

import time
import uuid

import numpy as np

from rebot_grasp.core.types import RGBDFrame


class OrbbecCamera:
    def __init__(self, width=1280, height=720, fps=30):
        self.width, self.height, self.fps = width, height, fps
        self._pipeline = None
        self.color_distortion = None

    def open(self):
        if self._pipeline is not None:
            raise RuntimeError("Camera is already open")
        from pyorbbecsdk import OBError
        try:
            self._open_pipeline()
        except OBError as exc:
            raise RuntimeError(f"Could not start Orbbec camera: {exc}") from exc

    def _open_pipeline(self):
        from pyorbbecsdk import Config, OBAlignMode, OBError, OBFormat, OBFrameAggregateOutputMode, OBSensorType, Pipeline

        pipeline, config = Pipeline(), Config()
        colors = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color = None
        depth = None
        for fmt in (OBFormat.RGB, OBFormat.MJPG):
            try:
                profile = colors.get_video_stream_profile(self.width, self.height, fmt, self.fps)
            except (OBError, RuntimeError):
                continue
            compatible = pipeline.get_d2c_depth_profile_list(profile, OBAlignMode.HW_MODE)
            # Older SDKs raise an OBError instead of StopIteration at the end of this list.
            matching = []
            for index in range(len(compatible)):
                entry = compatible[index].as_video_stream_profile()
                if entry.get_fps() == self.fps and entry.get_format() == OBFormat.Y16:
                    matching.append(entry)
            if matching:
                color = profile
                depth = next(
                    (entry for entry in matching if (entry.get_width(), entry.get_height()) == (self.width, self.height)),
                    matching[0],
                )
                break
        if color is None:
            raise RuntimeError(
                f"No hardware-aligned Orbbec RGB-D profile for {self.width}x{self.height}@{self.fps}. "
                "Select a supported color resolution and frame rate."
            )
        config.enable_stream(color)
        config.enable_stream(depth)
        config.set_align_mode(OBAlignMode.HW_MODE)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync()
        try:
            pipeline.start(config)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise
        self._pipeline = pipeline

    def capture(self):
        if self._pipeline is None:
            raise RuntimeError("Call open() before capture()")
        import cv2
        from pyorbbecsdk import OBError, OBFormat
        try:
            frames = self._pipeline.wait_for_frames(3000)
        except OBError as exc:
            raise RuntimeError(f"Could not read Orbbec camera: {exc}") from exc
        if frames is None:
            raise RuntimeError("Timed out waiting for Orbbec frames")
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if color is None or depth is None:
            raise RuntimeError("Missing aligned Orbbec color/depth frame")
        color_profile = color.get_stream_profile().as_video_stream_profile()
        color_intr = color_profile.get_intrinsic()
        depth_intr = depth.get_stream_profile().as_video_stream_profile().get_intrinsic()
        color_params = [color_intr.fx, color_intr.fy, color_intr.cx, color_intr.cy]
        depth_params = [depth_intr.fx, depth_intr.fy, depth_intr.cx, depth_intr.cy]
        if not np.allclose(color_params, depth_params, rtol=0.002, atol=0.5):
            raise RuntimeError("Aligned Orbbec depth and color intrinsics disagree")
        fx, fy, cx, cy = color_params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion = color_profile.get_distortion()
        self.color_distortion = np.array([
            distortion.k1, distortion.k2, distortion.p1, distortion.p2,
            distortion.k3, distortion.k4, distortion.k5, distortion.k6,
        ], dtype=float)
        raw = np.frombuffer(color.get_data(), dtype=np.uint8)
        if color.get_format() == OBFormat.MJPG:
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        elif color.get_format() == OBFormat.RGB:
            bgr = cv2.cvtColor(raw.reshape(color.get_height(), color.get_width(), 3), cv2.COLOR_RGB2BGR)
        else:
            raise RuntimeError("Unsupported Orbbec color format")
        raw_depth = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width())
        # Orbbec get_depth_scale returns millimeters per raw unit.
        depth_m = raw_depth.astype(np.float32) * float(depth.get_depth_scale()) * 0.001
        return RGBDFrame(bgr, depth_m, K, uuid.uuid4().hex, time.monotonic())

    def close(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
