import numpy as np
import pytest

from rebot_grasp.core.types import GraspCandidate, GraspPlan
from rebot_grasp.pipeline.execution import execute_plan


@pytest.fixture
def plan():
    grasp = GraspCandidate(np.eye(4), 0.04, 0.9, "test", "frame")
    return GraspPlan(grasp, np.eye(4), np.eye(4), np.eye(4), 0.045)


class Robot:
    def __init__(self, fail_move=None):
        self.calls = []
        self.moves = 0
        self.fail_move = fail_move
    def reachable(self, pose):
        self.calls.append("ik")
        return True
    def open_gripper(self, width):
        self.calls.append("open")
    def move(self, pose, duration):
        self.moves += 1
        self.calls.append("move")
        return self.moves != self.fail_move
    def close_gripper(self):
        self.calls.append("close")
        return True


class Validator:
    def validate(self, robot, plan):
        return True


def test_dry_run_makes_zero_robot_calls(plan):
    robot = Robot()
    result = execute_plan(plan, robot)
    assert result.stage == "dry_run"
    assert not result.success
    assert robot.calls == []


def test_execution_without_path_validation_never_opens_gripper(plan):
    robot = Robot()
    assert execute_plan(plan, robot, dry_run=False).stage == "validation"
    assert robot.calls == []


def test_failed_approach_does_not_close_or_return_home(plan):
    robot = Robot(fail_move=2)
    result = execute_plan(plan, robot, dry_run=False, validator=Validator())
    assert result.stage == "grasp"
    assert robot.calls == ["ik", "ik", "ik", "open", "move", "move"]


def test_successful_execution_checks_all_three_poses(plan):
    robot = Robot()
    result = execute_plan(plan, robot, dry_run=False, validator=Validator())
    assert result.success
    assert robot.calls == ["ik", "ik", "ik", "open", "move", "move", "close", "move"]


@pytest.mark.parametrize("with_retreat", [False, True])
def test_b601_grasp_sequence_preserves_optional_graspnet_retreat(monkeypatch, with_retreat):
    from types import SimpleNamespace

    from rebot_grasp.robots import actions

    events = []
    def move(*pose, duration):
        events.append((pose, duration))
        return True

    def close():
        events.append("close")
        return True

    controller = SimpleNamespace(move_to_traj=move)
    gripper = SimpleNamespace(open_gripper=lambda: events.append("open"), grasp=close)
    monkeypatch.setattr(actions, "wait_motion", lambda *_: None)
    monkeypatch.setattr(actions, "move_ready", lambda *_: events.append("ready"))
    pregrasp, grasp, retreat = (1.0,) * 6, (2.0,) * 6, (3.0,) * 6

    assert actions.execute_grasp(controller, gripper, grasp, pregrasp, {}, False,
                                 retreat6d=retreat if with_retreat else None)
    expected = ["open", (pregrasp, 2.0), (grasp, 1.5), "close"]
    if with_retreat:
        expected.append((retreat, 1.5))
    assert events == expected + ["ready"]


@pytest.mark.parametrize(("encoder", "fault"), [(4.0, 0), (float("nan"), 0), (4.9, 1)])
def test_unconfirmed_rs_opening_stops_before_approach(encoder, fault):
    from types import SimpleNamespace
    from rebot_grasp.robots import actions
    from rebot_grasp.robots.gripper_driver import GraspDriver

    driver = GraspDriver.__new__(GraspDriver)
    driver.open_gripper = lambda: None
    driver._position_done = lambda: True
    driver._open_soft_limit, driver._arrive_tol = 4.9, 0.12
    driver._gripper_group = SimpleNamespace(_jcfgs=[SimpleNamespace(vendor="robstride")])
    driver._gripper_motor = SimpleNamespace(
        robstride_get_param_f32=lambda *_: encoder,
        robstride_get_fault_report=lambda: (fault, 0),
        get_state=lambda: SimpleNamespace(fault_code=0),
    )
    robot = SimpleNamespace(move_to_traj=lambda *a, **kw: pytest.fail("Unexpected arm motion"))
    with pytest.raises(RuntimeError, match="Gripper"):
        actions.execute_grasp(robot, driver, (0,) * 6, (0,) * 6, {}, False, verify_open=True)
