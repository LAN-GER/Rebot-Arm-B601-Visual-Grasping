from dataclasses import replace

import numpy as np
import pytest

from rebot_grasp.core.types import GraspCandidate
from rebot_grasp.grasping.backends.obb import OBBBackend
from rebot_grasp.pipeline.capture import capture_stationary
from rebot_grasp.pipeline.planning import plan_capture


def test_obb_offline_chain_generates_plan(config, scene):
    capture, target = scene
    config = replace(config, planning={**config.planning, "T_grasp_tcp": np.eye(4)})
    result = plan_capture(capture, [target], OBBBackend(**config.backend_options), config)
    assert result.status == "planned_not_executed"
    assert len(result.plans) == 1
    candidate = result.candidates[0]
    assert 0.025 < candidate.width_m < 0.045
    assert np.isclose(candidate.T_camera_grasp[2, 3], 0.465, atol=1e-4)
    assert result.plans[0].T_base_pregrasp[2, 3] > result.plans[0].T_base_grasp[2, 3]


def test_no_target_does_not_call_backend(config, scene):
    class NeverCalled:
        def predict(self, *args):
            raise AssertionError("Must not run on a different object")
    result = plan_capture(scene[0], [scene[1]], NeverCalled(), replace(config, target_class="missing"))
    assert result.status == "target_not_found"


def test_gripper_opening_margin_filters_otherwise_valid_grasp(config, scene):
    capture, target = scene
    class WideBackend:
        def predict(self, *args):
            return [GraspCandidate(np.eye(4), 0.088, 0.9, "test", capture.frame.frame_id)]
    result = plan_capture(capture, [target], WideBackend(), config)
    assert result.status == "no_valid_grasp"


def test_wrong_frame_candidate_rejected_even_without_calibration(config, scene):
    class StaleBackend:
        def predict(self, *args):
            return [GraspCandidate(np.eye(4), 0.04, 0.9, "test", "stale")]
    with pytest.raises(ValueError, match="another frame"):
        plan_capture(scene[0], [scene[1]], StaleBackend(), config)


def test_capture_rejects_robot_motion(scene):
    class Camera:
        def capture(self):
            return scene[0].frame
    class Robot:
        calls = 0
        def tcp_pose(self):
            pose = np.eye(4)
            pose[0, 3] = self.calls * 0.01
            self.calls += 1
            return pose
    with pytest.raises(RuntimeError, match="moved"):
        capture_stationary(Camera(), Robot(), np.eye(4))
