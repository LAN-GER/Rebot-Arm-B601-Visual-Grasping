import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rebot_grasp.cameras.orbbec import OrbbecCamera
from rebot_grasp.cameras.realsense import RealSenseCamera


class Frame:
    def __init__(self, data, fmt=None):
        self.data, self.fmt = data, fmt
    def get_data(self):
        return self.data
    def get_height(self):
        return self.data.shape[0]
    def get_width(self):
        return self.data.shape[1]
    def get_format(self):
        return self.fmt
    def get_depth_scale(self):
        return 0.25  # Orbbec millimeters per raw unit, not meters.

    def get_stream_profile(self):
        return self

    def as_video_stream_profile(self):
        return self

    def get_intrinsic(self):
        return SimpleNamespace(fx=100.0, fy=100.0, cx=1.0, cy=1.0)

    def get_distortion(self):
        return SimpleNamespace(k1=0.1, k2=0.02, p1=0.003, p2=0.004, k3=0.05, k4=0.06, k5=0.07, k6=0.08)


def test_realsense_converts_raw_depth_to_meters_once():
    camera = RealSenseCamera()
    color = Frame(np.zeros((2, 3, 3), dtype=np.uint8))
    intr = SimpleNamespace(fx=100, fy=100, ppx=1, ppy=1)
    color.profile = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr))
    depth = Frame(np.full((2, 3), 500, dtype=np.uint16))
    frames = SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth)
    camera._pipeline = SimpleNamespace(wait_for_frames=lambda timeout: frames)
    camera._align = SimpleNamespace(process=lambda frames: frames)
    camera._scale_m = 0.001
    assert np.allclose(camera.capture().depth_m, 0.5)


def test_orbbec_respects_depth_scale_and_rejects_misalignment(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", SimpleNamespace(
        OBError=RuntimeError, OBFormat=SimpleNamespace(RGB=1, MJPG=2),
    ))
    camera = OrbbecCamera()
    color = Frame(np.zeros((2, 3, 3), dtype=np.uint8), fmt=1)
    depth = Frame(np.full((2, 3), 2000, dtype=np.uint16))
    frames = SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth)
    camera._pipeline = SimpleNamespace(wait_for_frames=lambda timeout: frames)
    assert np.allclose(camera.capture().depth_m, 0.5)
    assert camera.color_distortion.tolist() == [0.1, 0.02, 0.003, 0.004, 0.05, 0.06, 0.07, 0.08]
    depth.get_intrinsic = lambda: SimpleNamespace(fx=200.0, fy=100.0, cx=1.0, cy=1.0)
    with pytest.raises(RuntimeError, match="intrinsics disagree"):
        camera.capture()
    del depth.get_intrinsic
    depth.data = np.ones((1, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match="aligned"):
        camera.capture()
