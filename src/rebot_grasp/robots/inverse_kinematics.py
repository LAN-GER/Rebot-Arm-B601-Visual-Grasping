"""Check a B601 Cartesian target against current joint feedback."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reBotArm_control_py.actuator import RebotArm


class IkChecker:
    def __init__(self, arm: RebotArm) -> None:
        from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model
        from reBotArm_control_py.kinematics.inverse_kinematics import IKParams

        self._arm = arm
        self._arm_group = arm.groups.get("arm")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        self._n = self._arm_group.num_joints
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._params = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)

    def check(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> tuple[bool, float]:
        from reBotArm_control_py.kinematics import pad_q_for_model, pos_rot_to_se3, solve_ik

        q_now = self._arm.get_state(request_feedback=False)[0][: self._n]
        q_init = pad_q_for_model(self._model, q_now, self._n)
        target = pos_rot_to_se3(np.array([x, y, z], dtype=np.float64), roll=roll, pitch=pitch, yaw=yaw)
        result = solve_ik(
            self._model,
            self._data,
            self._end_frame_id,
            target,
            q_init,
            self._params,
            controlled_joints=self._n,
        )
        return bool(result.success), float(result.error)
