import numpy as np
import pytest

from rebot_grasp.core.types import RGBDFrame, Target
from rebot_grasp.grasping.backends.anygrasp import AnyGraspBackend
from rebot_grasp.perception.pointcloud import from_rgbd


class Detector:
    def __init__(self, records=None):
        self.calls = []
        self.records = records or []

    def predict(self, points, options):
        self.calls.append((points, options))
        return self.records


def small_frame():
    depth = np.array([[0.5, np.nan, 0.4], [0.0, 0.3, 2.0]], dtype=np.float32)
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    color[0, 0] = [10, 20, 30]
    return RGBDFrame(color, depth, np.eye(3), "test", 0)


def test_pointcloud_keeps_environment_and_aligns_target_mask():
    frame = small_frame()
    target = Target("object", [[1, 0, 0], [0, 1, 0]])
    cloud = from_rgbd(frame, target, 0.05, 1)
    assert cloud.points.dtype == np.float32
    assert np.allclose(cloud.points, [[0, 0, 0.5], [0.8, 0, 0.4], [0.3, 0.3, 0.3]])
    assert cloud.target_mask.tolist() == [True, False, True]
    assert np.allclose(cloud.colors_rgb[0], [30 / 255, 20 / 255, 10 / 255])


def test_empty_target_never_disables_sdk_region_steering():
    detector = Detector()
    backend = AnyGraspBackend(detector)
    frame = small_frame()
    assert backend.predict(frame, Target("absent", np.zeros((2, 3)))) == []
    # Target exists only on a pixel with invalid depth.
    assert backend.predict(frame, Target("invalid-depth", [[0, 1, 0], [0, 0, 0]])) == []
    assert detector.calls == []


def test_anygrasp_depth_moves_origin_once_and_keeps_rotation():
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    detector = Detector([dict(translation=[0.1, 0.2, 0.3], rotation=rotation, depth=0.02, width=0.04, score=0.9)])
    backend = AnyGraspBackend(detector)
    result, = backend.predict(small_frame(), None)
    assert np.allclose(result.T_camera_grasp[:3, 3], [0.1, 0.2, 0.32])
    assert np.allclose(result.T_camera_grasp[:3, :3], rotation)
    assert detector.calls[0][1]["region_steering"] is None
    assert detector.calls[0][1]["collision_detection"] is True


def test_bad_sdk_rotation_is_rejected():
    detector = Detector([dict(translation=[0, 0, 0.5], rotation=np.diag([1, 1, -1]), depth=0.02, width=0.04, score=0.9)])
    assert AnyGraspBackend(detector).predict(small_frame(), None) == []


def test_extra_insertion_past_sdk_tip_and_workspace_check():
    # Tilt +X so confusing the approach direction with camera Z changes the result.
    rotation = np.array([[0.8, 0, -0.6], [0, 1, 0], [0.6, 0, 0.8]])
    detector = Detector([dict(translation=[0, 0, .30], rotation=rotation, depth=.02,
                              width=.04, score=.9)])
    backend = AnyGraspBackend(detector, insertion_depth_m=.005)
    candidate, = backend.predict(small_frame(), None)
    np.testing.assert_allclose(candidate.T_camera_grasp[:3, 3], [.020, 0, .315])
    workspace = {"min_m": [-1, -1, 0], "max_m": [1, 1, .314]}
    assert AnyGraspBackend(detector, workspace_camera=workspace).predict(small_frame(), None)
    bounded = AnyGraspBackend(detector, insertion_depth_m=.005, workspace_camera=workspace)
    assert bounded.predict(small_frame(), None) == []


@pytest.mark.parametrize("invalid", [-.005, float("inf"), float("nan"), True])
def test_invalid_extra_insertion(invalid):
    with pytest.raises(ValueError, match="insertion_depth_m"):
        AnyGraspBackend(Detector(), insertion_depth_m=invalid)
