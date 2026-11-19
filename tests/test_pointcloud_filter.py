import numpy as np
import pytest

from rebot_grasp.perception.pointcloud import PointCloud, filter_outliers
from rebot_grasp.core.types import RGBDFrame, Target
from rebot_grasp.grasping.backends.anygrasp import AnyGraspBackend


def test_isolated_point_removed_with_color_and_mask_alignment():
    xy = np.indices((10, 10)).reshape(2, -1).T * .001
    points = np.vstack([np.c_[xy, np.full(100, .2)], [1, 1, 1]]).astype(np.float32)
    colors = np.arange(303).reshape(101, 3) / 303
    mask = np.arange(101) % 2 == 0
    filtered = filter_outliers(PointCloud(points, colors, mask))
    assert len(filtered.points) == 100
    np.testing.assert_array_equal(filtered.points, points[:100])
    np.testing.assert_array_equal(filtered.colors_rgb, colors[:100])
    np.testing.assert_array_equal(filtered.target_mask, mask[:100])


def test_filtering_away_only_target_never_calls_sdk():
    class Detector:
        def predict(self, *args):
            pytest.fail('Empty target must not become full-scene inference')
    depth = np.full((20, 20), .2, dtype=np.float32)
    depth[10, 10] = .9
    frame = RGBDFrame(np.zeros((20, 20, 3), np.uint8), depth,
                      np.array([[200, 0, 10], [0, 200, 10], [0, 0, 1]]), 'filter', 0)
    mask = np.zeros(depth.shape, bool)
    mask[10, 10] = True
    backend = AnyGraspBackend(Detector(), outlier_filter={'nb_neighbors': 20, 'std_ratio': 2.})
    assert backend.predict(frame, Target('banana', mask)) == []


def test_small_and_empty_clouds_are_preserved():
    for n in (0, 2):
        cloud = PointCloud(np.zeros((n, 3)), np.zeros((n, 3)), None)
        assert filter_outliers(cloud) is cloud
