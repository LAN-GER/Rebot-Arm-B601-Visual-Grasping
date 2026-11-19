"""B601 adapters and an explicit session using the preserved B601 control flow."""

import numpy as np

from rebot_grasp.core.types import checked_transform
from rebot_grasp.geometry.transforms import pose6d


class B601Robot:
    def __init__(self, controller, gripper, ik_checker):
        self.controller = controller
        self.gripper = gripper
        self.ik_checker = ik_checker

    def tcp_pose(self):
        return checked_transform(self.gripper.get_tcp_pose())

    def reachable(self, T_base_tcp):
        success, _ = self.ik_checker.check(*pose6d(T_base_tcp))
        return bool(success)

    def move(self, T_base_tcp, duration_s):
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Motion duration must be positive")
        if not self.controller.move_to_traj(*pose6d(T_base_tcp), duration=duration_s):
            return False
        thread = getattr(self.controller, "_send_thread", None)
        if thread is None:
            raise RuntimeError("SDK provides no observable motion thread; cannot confirm completion")
        thread.join(timeout=duration_s + 3)
        if thread.is_alive():
            return False
        # Thread completion alone does not establish arrival: verify pose feedback.
        actual = self.tcp_pose()
        target = checked_transform(T_base_tcp)
        angle = np.arccos(np.clip((np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))
        return bool(np.linalg.norm(actual[:3, 3] - target[:3, 3]) < 0.005 and angle < 0.05)

    def open_gripper(self, width_m):
        if not 0 < width_m <= self.gripper.MAX_DISTANCE_M:
            raise ValueError("Requested opening exceeds gripper range")
        self.gripper.open_gripper(distance_m=width_m)

    def close_gripper(self):
        return bool(self.gripper.grasp())


class B601Session:
    """Use the original B601 controller, gripper driver and OBB action sequence."""

    def __init__(self, robot_config, *, verify_open=False):
        from rebot_grasp.robots.gripper_driver import ensure_rebot_sdk_in_syspath
        ensure_rebot_sdk_in_syspath(robot_config.get("repo_root"))
        from rebot_grasp.robots import actions

        self.actions = actions
        self.verify_open = verify_open
        self.config = robot_config
        self.arm = None
        self.controller = None
        self.gripper = None
        self.started = False

    def start(self):
        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.controllers import RebotArmEndPose
        from rebot_grasp.robots.gripper_driver import GraspDriver, selected_arm_config

        selected = selected_arm_config(self.config.get("repo_root"))
        self.arm = RebotArm()
        self.controller = RebotArmEndPose(self.arm, arm_control_mode=selected.controller_mode)
        self.gripper = GraspDriver(
            self.arm, self.controller, gripper_config=self.config.get("gripper"),
            repo_root=self.config.get("repo_root"),
        )
        self.gripper.start()
        self.started = True
        self.actions.move_ready(self.controller, self.config["ready_pose"])

    def tcp_pose(self):
        return checked_transform(self.gripper.get_tcp_pose())

    def reachable(self, pose):
        from reBotArm_control_py.kinematics import pad_q_for_model, pos_rot_to_se3
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik

        model = self.controller._model
        q = self.arm.get_state(request_feedback=False)[0]
        q = pad_q_for_model(model, q, self.controller._n)
        x, y, z, roll, pitch, yaw = pose6d(pose)
        target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)
        result = solve_ik(model, model.createData(), self.controller._end_frame_id,
                          target, q, self.controller._ik_solver_params,
                          controlled_joints=self.controller._n)
        return bool(result.success)

    def execute(self, plan):
        return self.actions.execute_grasp(
            self.controller, self.gripper, pose6d(plan.T_base_grasp),
            pose6d(plan.T_base_pregrasp), self.config["ready_pose"], dry_run=False,
            verify_open=self.verify_open,
        )

    def close(self):
        try:
            if self.started and self.controller._running:
                try:
                    self.gripper.release_gripper()
                finally:
                    self.controller.end()
        finally:
            if self.arm is not None:
                self.arm.disconnect()
            self.started = False
