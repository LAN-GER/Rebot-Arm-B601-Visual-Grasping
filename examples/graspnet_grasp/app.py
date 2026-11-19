"""
Visual grasping demo based on GraspNet-baseline and YOLO

Keys:
  G/Space: run GraspNet on the current RGB-D frame and execute a grasp.
  R: resume live preview.
  Q/Esc: release, home, and exit.

Usage:
    bash examples/graspnet_grasp/run.sh --dry-run
    bash examples/graspnet_grasp/run.sh --target-class cup
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

from rebot_grasp.paths import PROJECT_ROOT
from rebot_grasp.robots.actions import execute_grasp, move_ready
from rebot_grasp.robots.inverse_kinematics import IkChecker


from rebot_grasp.cameras.streams import make_camera
from rebot_grasp.robots.gripper_driver import GRIPPER_MAX_DISTANCE_M, GraspDriver, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from rebot_grasp.pipeline import graspnet_camera
from rebot_grasp.visualization import graspnet as graspnet_view
from rebot_grasp.geometry.transforms import compose_cam_to_base_transform
from rebot_grasp.cameras.settings import configure_camera
from rebot_grasp.config import load_example_config
from rebot_grasp.calibration.hand_eye import load_hand_eye
from rebot_grasp.geometry.grasp_frames import (
    canonicalize_parallel_gripper_tcp_rotation,
    graspnet_rotation_to_rebot_tcp_rotation,
    rotation_matrix_to_euler_zyx,
)
from rebot_grasp.perception.detections import (
    YoloDetection,
    detect_objects,
    load_yolo as load_yolo_from_config,
)
if TYPE_CHECKING:
    from graspnetAPI import Grasp, GraspGroup


def _print_grasp(grasp: Grasp) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix))
    print("\n[G] Best GraspNet grasp:")
    print(f"  score={grasp.score:.4f} width={grasp.width:.4f} height={grasp.height:.4f} depth={grasp.depth:.4f}")
    print(f"  position_xyz={grasp.translation.tolist()}")
    print(f"  graspnet_rpy={rotation_matrix_to_euler_zyx(grasp.rotation_matrix).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _rank_grasps(grasps: GraspGroup) -> GraspGroup:
    ranked = graspnet_camera.copy_grasp_group(grasps)
    try:
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS skipped: {exc}")
    ranked.sort_by_score()
    return ranked


def _pose_z_ok(pose6d: tuple[float, ...], min_z: float) -> bool:
    return float(pose6d[2]) >= float(min_z)


def _select_executable_grasp(
    ik_checker: IkChecker,
    grasps: GraspGroup,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float,
    min_base_z_m: float,
) -> Optional[tuple[Grasp, tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    ranked = _rank_grasps(grasps)
    skipped_low = 0
    skipped_ik = 0
    worst_err = 0.0

    for idx in range(len(ranked)):
        grasp = ranked[idx]
        grasp6d, pre6d, retreat6d = graspnet_camera.grasp_to_base_poses(
            grasp,
            T_cam2base,
            pregrasp_offset_m,
            retreat_offset_m,
            insertion_depth_m,
        )
        if not (_pose_z_ok(pre6d, min_base_z_m) and _pose_z_ok(grasp6d, min_base_z_m)):
            skipped_low += 1
            continue

        pre_ok, pre_err = ik_checker.check(*pre6d)
        grasp_ok, grasp_err = ik_checker.check(*grasp6d) if pre_ok else (False, pre_err)
        worst_err = max(worst_err, pre_err, grasp_err)
        if pre_ok and grasp_ok:
            print(f"[G] Executable rank={idx + 1}/{len(ranked)} score={grasp.score:.4f}")
            if skipped_low or skipped_ik:
                print(f"[G] Skipped low_z={skipped_low} ik_fail={skipped_ik}")
            return grasp, grasp6d, pre6d, retreat6d
        skipped_ik += 1

    print(f"[G] No IK-reachable grasp: low_z={skipped_low} ik_fail={skipped_ik} max_err={worst_err:.4f}")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraspNet robot grasp demo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true", help="skip the grasp sequence; the arm still moves to ready on startup and homes on exit")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2", "orbbec_gemini336"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-point", type=int, default=None)
    parser.add_argument("--num-view", type=int, default=None)
    parser.add_argument("--collision-thresh", type=float, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--min-depth", type=float, default=None, help="meters")
    parser.add_argument("--max-depth", type=float, default=None, help="meters")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--extra-yolo-class", action="append", default=[], help="add open-vocabulary YOLO class")
    parser.add_argument("--target-margin-px", type=int, default=None)
    parser.add_argument("--target-expand-ratio", type=float, default=None, help="YOLO bbox expansion ratio")
    parser.add_argument("--no-yolo", action="store_true", help="disable YOLO and use full-scene GraspNet")
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--infer-every-live", type=int, default=None)
    parser.add_argument("--pregrasp-offset", type=float, default=None, help="meters")
    parser.add_argument("--retreat-offset", type=float, default=None, help="meters")
    parser.add_argument("--min-base-z", type=float, default=None, help="minimum executable TCP z in base frame, meters")
    parser.add_argument("--no-open3d", action="store_true", help="do not open Open3D after inference")
    parser.add_argument(
        "--open3d-grasps",
        choices=("final", "bbox", "pre-bbox"),
        default="final",
        help="Open3D grasp set: final, bbox, or pre-bbox",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = configure_camera(load_example_config(Path(args.config)), args)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})
    pregrasp_offset_m = float(args.pregrasp_offset if args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08))
    retreat_offset_m = float(args.retreat_offset if args.retreat_offset is not None else pregrasp_offset_m)
    insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
    min_base_z_m = float(args.min_base_z if args.min_base_z is not None else grasp_cfg.get("min_base_z_m", 0.03))
    graspnet_cfg = cfg.get("graspnet", {})
    if args.target_class is None:
        args.target_class = graspnet_cfg.get("target_class")
    voxel_size = args.voxel_size if args.voxel_size is not None else graspnet_cfg.get("voxel_size", 0.01)
    min_depth_m = float(args.min_depth if args.min_depth is not None else graspnet_cfg.get("min_depth", 0.05))
    max_depth_m = float(args.max_depth if args.max_depth is not None else graspnet_cfg.get("max_depth", 2.0))
    num_point = int(args.num_point if args.num_point is not None else graspnet_cfg.get("num_point", 20000))
    collision_thresh = float(
        args.collision_thresh if args.collision_thresh is not None else graspnet_cfg.get("collision_thresh", 0.01)
    )
    target_expand_ratio = float(
        args.target_expand_ratio
        if args.target_expand_ratio is not None
        else graspnet_cfg.get("target_expand_ratio", 1.0)
    )

    cam_cfg = cfg["camera"]
    print(f"=== Init camera: {cam_cfg['type']} {cam_cfg.get('color_width')}x{cam_cfg.get('color_height')}@{cam_cfg.get('fps')} ===")
    cam = make_camera(cfg)

    last_detections: list[YoloDetection] = []
    selected_target: Optional[Any] = None
    last_target_status = "target detector warming up..."
    status = "warming up camera..."
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0
    window_name = "Main - GraspNet Grasp"
    top_k = int(cfg.get("graspnet", {}).get("top_k", 50))
    vis: Optional[graspnet_view.Open3DGraspWindow] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cam_cfg.get("color_width", 1280)), int(cam_cfg.get("color_height", 720)))
    print("\n[Keys] G/SPACE=grasp  R=resume  Q/ESC=quit\n")

    rebotarm: Optional[RebotArm] = None
    controller: Optional[RebotArmEndPose] = None
    grasp_driver: Optional[GraspDriver] = None
    ik_checker: Optional[IkChecker] = None
    T_hand_eye: Optional[np.ndarray] = None
    robot_ready = False

    try:
        cam.open()
        cam.warm_up(args.warmup)
        K = cam.K.astype(np.float64)
        print("Camera intrinsics:")
        print(K)

        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        print("=== Load models ===")
        yolo_model, yolo_opts = load_yolo_from_config(
            cfg,
            project_root=PROJECT_ROOT,
            no_yolo=args.no_yolo,
            model_override=args.yolo_model,
            device_override=args.yolo_device,
            conf_override=args.yolo_conf,
            iou_override=args.yolo_iou,
            infer_every_override=args.infer_every_live,
            extra_classes=args.extra_yolo_class,
        )
        last_target_status = "YOLO disabled: full-scene GraspNet" if yolo_model is None else "target detector warming up..."
        net = graspnet_camera.build_net(
            args.checkpoint or graspnet_cfg.get("checkpoint", "checkpoint-rs.tar"),
            args.num_view if args.num_view is not None else graspnet_cfg.get("num_view", 300),
            sdk_root=graspnet_cfg.get("sdk_root", graspnet_camera.GRASPNET_ROOT),
        )

        print("=== Init robot ===")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        ik_checker = IkChecker(rebotarm)
        print(f"[Robot] mode: {selected.controller_mode}")

        print("[Robot] Move ready")
        move_ready(controller, ready_cfg)

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    selected_target = graspnet_camera.select_target(last_detections, args.target_class)
                    last_target_status = graspnet_camera.target_status_text(selected_target, last_detections, args.target_class)
                except Exception as exc:
                    last_detections = []
                    selected_target = None
                    last_target_status = f"YOLO failed: {exc}"

            if frozen and last_display is not None:
                display = last_display.copy()
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = graspnet_view.draw_detections_overlay(color_bgr, last_detections, selected_target, args.target_class)
                display = graspnet_view.draw_status(
                    display_base,
                    f"LIVE {fps_value:.1f}fps | {status}",
                    last_target_status,
                    title="Main - GraspNet Grasp",
                )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                status = "live preview"
                continue

            if key in (ord("g"), ord("G"), ord(" ")):
                print("\n[G] Capture and run GraspNet")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                try:
                    result = graspnet_camera.infer_frame(
                        net,
                        snap_color,
                        snap_depth,
                        K,
                        num_point=num_point,
                        min_depth=min_depth_m,
                        max_depth=max_depth_m,
                        collision_thresh=collision_thresh,
                        voxel_size=voxel_size,
                        yolo_model=yolo_model,
                        yolo_opts=yolo_opts,
                        target_class=args.target_class,
                        target_margin_px=int(
                            args.target_margin_px
                            if args.target_margin_px is not None
                            else graspnet_cfg.get("target_margin_px", 12)
                        ),
                        target_expand_ratio=target_expand_ratio,
                        max_grasp_width_m=GRIPPER_MAX_DISTANCE_M,
                    )
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(f"[G] {status}")
                    continue

                status = result.status
                last_target_status = result.target_status
                last_detections = result.detections
                selected_target = result.selected_target
                vis_grasps = graspnet_view.visualization_grasps(result, args.open3d_grasps)

                print(f"[G] {status}")
                if not args.no_open3d:
                    try:
                        if vis is None:
                            vis = graspnet_view.Open3DGraspWindow("GraspNet Grasps", top_k)
                        vis.update(result.o3d_cloud, vis_grasps)
                        print(f"[G] Open3D {args.open3d_grasps} candidates={len(vis_grasps)}")
                    except Exception as exc:
                        print(f"[G] Open3D failed: {exc}")
                        if vis is not None:
                            vis.close()
                            vis = None

                if result.best is None:
                    print("[G] No valid GraspNet grasp")
                    continue
                frozen = True
                display_base = snap_color
                if yolo_model is not None:
                    display_base = graspnet_view.draw_detections_overlay(snap_color, last_detections, selected_target, args.target_class)
                snap_display = graspnet_view.draw_status(
                    display_base,
                    f"SNAPSHOT | {status}",
                    last_target_status,
                    frozen=True,
                    title="Main - GraspNet Grasp",
                )
                last_display = snap_display

                if T_hand_eye is None:
                    graspnet_view.draw_best_grasp_projection(snap_display, result.best, K)
                    last_display = snap_display
                    print("[G] Hand-eye calibration unavailable")
                    continue

                T_cam2base = compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)
                selected = _select_executable_grasp(
                    ik_checker,
                    result.grasps,
                    T_cam2base,
                    pregrasp_offset_m,
                    retreat_offset_m,
                    insertion_depth_m,
                    min_base_z_m,
                )
                if selected is None:
                    print(f"[G] No IK-reachable grasp above min_base_z={min_base_z_m:.3f}m ")
                    continue
                best, grasp6d, pre6d, retreat6d = selected

                _print_grasp(best)
                graspnet_view.draw_best_grasp_projection(snap_display, best, K)
                last_display = snap_display

                execute_grasp(
                    controller,
                    grasp_driver,
                    grasp6d,
                    pre6d,
                    ready_cfg,
                    retreat6d=retreat6d,
                    dry_run=args.dry_run,
                )

            if vis is not None and not vis.poll():
                vis.close()
                vis = None

    finally:
        print("\n[Exit] Release gripper and home")
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            cam.close()
        except Exception:
            pass
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
