from rebot_grasp.core.interfaces import Robot
from rebot_grasp.core.types import GraspPlan


def select_executable(plans: list[GraspPlan], robot: Robot) -> GraspPlan | None:
    for plan in plans:
        if all(robot.reachable(pose) for pose in (plan.T_base_pregrasp, plan.T_base_grasp, plan.T_base_retreat)):
            return plan
    return None
