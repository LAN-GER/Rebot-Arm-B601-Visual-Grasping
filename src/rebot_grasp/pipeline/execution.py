"""Execution is opt-in, separately callable, and never exposed by preview CLI."""

from rebot_grasp.core.interfaces import PathValidator, Robot
from rebot_grasp.core.types import ExecutionResult, GraspPlan


def execute_plan(plan: GraspPlan, robot: Robot | None = None, *, dry_run: bool = True,
                 validator: PathValidator | None = None) -> ExecutionResult:
    # Must precede all hardware access, even state reads and IK checks.
    if dry_run:
        return ExecutionResult(False, "dry_run", "Plan inspected; no robot methods called")
    if robot is None or validator is None:
        return ExecutionResult(False, "validation", "Robot and current-scene path validator are required")
    stage = "validation"
    try:
        poses = (plan.T_base_pregrasp, plan.T_base_grasp, plan.T_base_retreat)
        if not all(robot.reachable(pose) for pose in poses) or not validator.validate(robot, plan):
            return ExecutionResult(False, stage, "Reachability or path validation failed")
        stage = "open_gripper"
        robot.open_gripper(plan.opening_m)
        for stage, pose in (("pregrasp", plan.T_base_pregrasp), ("grasp", plan.T_base_grasp)):
            if not robot.move(pose, 2.0):
                return ExecutionResult(False, stage, "Motion failed; no further motion requested")
        stage = "close_gripper"
        held = robot.close_gripper()
        stage = "retreat"
        if not robot.move(plan.T_base_retreat, 2.0):
            return ExecutionResult(False, stage, "Retreat failed; automatic return disabled")
        return ExecutionResult(held, "complete", "Object contact detected" if held else "No object contact detected")
    except Exception as exc:
        return ExecutionResult(False, stage, f"{type(exc).__name__}: {exc}")
