"""B601 trajectory and parallel-gripper actions shared by RGB-D applications."""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reBotArm_control_py.controllers import RebotArmEndPose
    from rebot_grasp.robots.gripper_driver import GraspDriver


def wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    wait_motion(controller, duration)


def execute_grasp(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
    *, verify_open=False, retreat6d: tuple[float, ...] | None = None,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Grasp] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print("[Grasp] dry run; skip motion")
        return False

    print("[Grasp] Open gripper")
    grasp_driver.open_gripper()
    if verify_open:
        grasp_driver.confirm_open()

    print("[Grasp] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Grasp] Pregrasp IK failed")
        return False
    wait_motion(controller, 2.0)

    print("[Grasp] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Grasp] Grasp IK failed")
        return False
    wait_motion(controller, 1.5)

    print("[Grasp] Closing")
    ok = grasp_driver.grasp()
    print("[Grasp] Holding object" if ok else "[Grasp] Empty grasp")

    if retreat6d is not None:
        print("[Grasp] Retreat")
        if controller.move_to_traj(*retreat6d, duration=1.5):
            wait_motion(controller, 1.5)

    print("[Grasp] Return ready")
    move_ready(controller, ready_cfg)
    return ok
