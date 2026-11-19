from dataclasses import replace

import numpy as np
import pytest

from rebot_grasp.core.types import Capture, GraspCandidate, checked_transform
from rebot_grasp.geometry.transforms import inverse, transform
from rebot_grasp.pipeline.planning import make_plans


def test_transform_composition_and_inverse():
    T = transform(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), [1, 2, 3])
    assert np.allclose(T @ inverse(T), np.eye(4))
    assert np.allclose(T @ [1, 0, 0, 1], [1, 3, 3, 1])


@pytest.mark.parametrize("rotation", [np.diag([1, 1, -1]), np.eye(3) * 2])
def test_reject_reflections_and_scaled_rotations(rotation):
    with pytest.raises(ValueError, match="proper rotation"):
        transform(rotation)


def test_plan_retreat_uses_grasp_axis_even_with_rotated_tcp(config, scene):
    capture, _ = scene
    capture = Capture(capture.frame, np.eye(4))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    config = replace(config, planning={**config.planning, "T_grasp_tcp": transform(rotation, [0.01, 0, 0])})
    grasp = GraspCandidate(transform(translation=[0.3, 0.1, 0.3]), 0.04, 0.8, "test", capture.frame.frame_id)
    plan, = make_plans(capture, [grasp], config)
    assert np.allclose(plan.T_base_grasp[:3, 3], [0.31, 0.1, 0.3])
    assert np.allclose(plan.T_base_pregrasp[:3, 3], [0.23, 0.1, 0.3])
    assert np.allclose(plan.T_base_retreat[:3, 3], [0.21, 0.1, 0.3])


def test_missing_tool_transform_does_not_default_to_identity(config, scene):
    capture, _ = scene
    grasp = GraspCandidate(np.eye(4), 0.04, 0.8, "test", capture.frame.frame_id)
    assert make_plans(capture, [grasp], config) == []


def test_workspace_checks_retreat_as_well_as_grasp(config, scene):
    capture, _ = scene
    capture = Capture(capture.frame, np.eye(4))
    config = replace(config, planning={**config.planning, "T_grasp_tcp": np.eye(4), "workspace_min_m": [0.0, -0.6, 0.0]})
    grasp = GraspCandidate(transform(translation=[0.09, 0, 0.2]), 0.04, 0.8, "test", capture.frame.frame_id)
    # Grasp and pregrasp are in bounds, but the 10 cm retreat is outside.
    assert make_plans(capture, [grasp], config) == []


def test_public_transform_is_copied_and_immutable():
    source = np.eye(4)
    stored = checked_transform(source)
    source[0, 3] = 5
    assert stored[0, 3] == 0
    with pytest.raises(ValueError):
        stored[0, 3] = 2
