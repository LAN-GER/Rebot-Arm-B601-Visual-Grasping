from types import SimpleNamespace

import numpy as np

from rebot_grasp.core.types import Target
from rebot_grasp.grasping.backends.graspnet import GraspNetBackend


def test_graspnet_samples_target_and_checks_collisions_on_scene(scene):
    capture, target = scene

    class Detector:
        def predict(self, sampled, scene_points, collision_thresh, voxel_size):
            assert sampled.shape == (32, 3)
            assert sampled.dtype == np.float32
            assert np.allclose(sampled[:, 2], 0.45)
            assert np.any(np.isclose(scene_points[:, 2], 0.60))
            rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            return [
                SimpleNamespace(translation=[0.2, 0, 0.45], rotation_matrix=rotation, depth=0.02, width=0.04, score=0.9),
                SimpleNamespace(translation=[0, 0, 0.45], rotation_matrix=rotation, depth=0.02, width=0.04, score=0.8),
            ]

    backend = GraspNetBackend(sdk_root="", checkpoint="", num_point=32, detector=Detector())
    candidate, = backend.predict(capture.frame, target)
    assert candidate.backend == "graspnet"
    assert np.allclose(candidate.T_camera_grasp[:3, 3], [0, 0, 0.47])


def test_graspnet_empty_target_does_not_call_detector(scene):
    capture, target = scene
    backend = GraspNetBackend(sdk_root="", checkpoint="", detector=object())
    empty_target = Target(target.label, np.zeros_like(target.mask))
    assert backend.predict(capture.frame, empty_target) == []
