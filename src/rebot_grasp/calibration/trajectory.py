"""Collect the SDK controller's original calibration setpoints for path validation."""
from types import SimpleNamespace

import numpy as np


def make_trajectory_planner(base, validation_urdf, plan_type, error_type):
    from reBotArm_control_py.controllers import RebotArmEndPose
    from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model

    class PlanningController(RebotArmEndPose):
        def _send_loop(self, duration):
            # move_to_traj has already populated _traj. This instance owns no devices.
            pass

    class CalibrationTrajectoryPlanner(base):
        def __init__(self, model, frame_id, config):
            # Include the mounted load when checking the allowable gravity torques.
            model_with_load = load_robot_model(str(validation_urdf))
            super().__init__(model_with_load, get_end_effector_frame_id(model_with_load), config)

        def plan(self, joints, xyzrpy, duration, entry=False):
            q = self.full_q(joints)
            standin = SimpleNamespace(groups={"arm": SimpleNamespace(num_joints=6)},
                                      has_gripper=False, arm_control_mode="mit",
                                      get_state=lambda: (q.copy(), None, None))
            ctrl = PlanningController(standin)
            ctrl._running = True
            if not ctrl.move_to_traj(*xyzrpy, duration=duration):
                raise error_type("TARGET_IK_FAILED")
            ctrl._send_thread.join()
            path = np.tile(q, (len(ctrl._traj), 1))
            path[:, :6] = ctrl._traj
            # The SDK sender sleeps duration / n between successive setpoints.
            times = np.arange(len(path) + 1) * (duration / len(path))
            held_path = np.vstack((path, path[-1]))
            try:
                report = self.validate(times, held_path, q)
            except error_type as exc:
                dt = duration / len(path)
                vel = np.diff(held_path[:, :6], axis=0) / dt
                acc = np.vstack((2 * vel[0], np.diff(vel, axis=0), -2 * vel[-1])) / dt
                raise error_type(
                    f"{exc}: target {np.asarray(xyzrpy).tolist()}, "
                    f"peak joint speed {np.max(np.abs(vel)):.3f} rad/s, "
                    f"peak joint acceleration {np.max(np.abs(acc)):.3f} rad/s^2"
                ) from exc
            report.update(points=len(path), duration_s=duration, source="sdk_cartesian_trajectory")
            # _publish uses the last time as the duration and sends exactly len(q) points.
            return plan_type(np.linspace(0., duration, len(path)), path, report)

    return CalibrationTrajectoryPlanner
