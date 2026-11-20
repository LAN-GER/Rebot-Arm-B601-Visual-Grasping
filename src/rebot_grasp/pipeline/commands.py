"""Application workflows used by the command-line examples."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from rebot_grasp.bootstrap import execution_calibration, make_backend, make_camera, make_detector
from rebot_grasp.cameras.offline import load_recording, save_recording
from rebot_grasp.cameras.synthetic import demo_capture
from rebot_grasp.config import load_config
from rebot_grasp.pipeline.capture import capture_rgbd
from rebot_grasp.pipeline.planning import plan_capture
from rebot_grasp.visualization.export import save_result


def run_obb_filtered(args) -> int:
    from rebot_grasp.pipeline import obb_filtered
    return _run_filtered(args, "obb_filtered", obb_filtered)


def run_anygrasp_filtered(args) -> int:
    from rebot_grasp.pipeline import anygrasp_filtered
    return _run_filtered(args, "anygrasp_filtered", anygrasp_filtered)


def _run_filtered(args, backend_name, workflow):
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path, backend_name)
    if args.insertion_depth_mm is not None:
        if not np.isfinite(args.insertion_depth_mm) or args.insertion_depth_mm < 0:
            raise ValueError('--insertion-depth-mm must be finite and nonnegative')
        config = replace(config, backend_options={**config.backend_options,
                         'insertion_depth_m': args.insertion_depth_mm / 1000.})
    if args.target_class is not None:
        if not args.target_class.strip():
            raise ValueError('--target-class cannot be empty')
        config = replace(config, target_class=args.target_class)
    if config.full_scene:
        raise ValueError('Filtered comparison requires a target')
    backend = make_backend(config)
    detector = make_detector(config)
    if args.input:
        args.output.mkdir(parents=True, exist_ok=False)
        from rebot_grasp.pipeline.live_session import save_session_settings
        save_session_settings(args.output, config, False)
        capture = load_recording(args.input)[0]
        try:
            # The CPU YOLO predictor can change CUDA visibility in this process.
            backend.start()
            detector.start()
            result, _ = workflow.infer_snapshot(capture, detector, backend, config, args.output)
        finally:
            backend.close()
        return 0 if result.candidates else 2
    hand_eye = None
    robot = None
    if args.execute:
        from rebot_grasp.bootstrap import make_robot
        hand_eye = execution_calibration(config, config_path)
        robot = make_robot(config)
    workflow.run_live(make_camera(config), backend, detector, config, args.output, robot, hand_eye,
                      show_diagnostics=not args.single_window)
    return 0


def run_camera_check(args) -> int:
    from rebot_grasp.visualization.segmentation import inspect_segmentation
    config = load_config(args.config, "anygrasp")
    if args.target_class is not None and not args.target_class.strip():
        raise ValueError("--target-class cannot be empty")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    detector = make_detector(config)
    detector.start()
    capture = (load_recording(args.input)[0] if args.input
               else capture_rgbd(make_camera(config), args.warmup))
    save_recording(args.output / "frame.npz", capture)
    targets = detector.detect(capture.frame)
    passed = inspect_segmentation(
        capture, targets, args.output,
        target_class=args.target_class if args.target_class is not None else config.target_class,
        outlier_filter=config.backend_options.get("outlier_filter"),
        workspace_camera=config.backend_options.get("workspace_camera"),
        gripper_exclusion=config.backend_options.get("gripper_exclusion"),
        pointcloud_quality=config.backend_options.get("pointcloud_quality"),
        min_depth_m=config.backend_options["min_depth_m"],
        max_depth_m=config.backend_options["max_depth_m"], view=args.view,
    )
    return 0 if passed else 2


def run_anygrasp_live(args) -> int:
    from rebot_grasp.bootstrap import make_robot
    from rebot_grasp.pipeline.anygrasp import run_live

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path, "anygrasp")
    if args.target_class is not None:
        if not args.target_class.strip():
            raise ValueError("--target-class cannot be empty")
        config = replace(config, target_class=args.target_class, full_scene=False)
    if args.full_scene:
        config = replace(config, full_scene=True, target_class=None)
    hand_eye = None
    use_robot = args.execute or args.diagnose_gripper
    if use_robot:
        hand_eye = execution_calibration(config, config_path)
    backend = make_backend(config)
    detector = None if config.full_scene else make_detector(config)
    robot = make_robot(config) if use_robot else None
    run_live(make_camera(config), backend, detector, config, args.output, robot, hand_eye,
             diagnose_gripper=args.diagnose_gripper)
    return 0


def run_record(args) -> int:
    if args.output.suffix != ".npz":
        raise ValueError("--output must be an .npz file")
    config = load_config(args.config)
    overrides = {
        key: value for key in ("type", "color_width", "color_height", "fps")
        if (value := getattr(args, key)) is not None
    }
    config = replace(config, camera={**config.camera, **overrides})
    capture = capture_rgbd(make_camera(config), args.warmup)
    height, width = capture.frame.depth_m.shape
    valid = np.isfinite(capture.frame.depth_m) & (capture.frame.depth_m > 0)
    if not valid.any():
        raise RuntimeError("Camera returned no valid depth")
    save_recording(args.output, capture)
    print(f"Saved {width}x{height} RGB-D to {args.output.resolve()} ({valid.mean():.1%} valid depth)")
    return 0


def run_preview(args) -> int:
    synthetic = args.command == "demo"
    config = load_config(args.config, "obb" if synthetic else args.backend)
    if synthetic:
        capture, target = demo_capture()
        targets = [target]
        config = replace(config, target_class=target.label,
                         planning={**config.planning, "T_grasp_tcp": np.eye(4)})
        save_recording(args.output / "frame.npz", capture, target)
    else:
        if args.target_class is not None:
            if not args.target_class.strip():
                raise ValueError("--target-class cannot be empty")
            config = replace(config, target_class=args.target_class, full_scene=False)
        if args.full_scene:
            if config.backend in ("obb", "obb_filtered", "anygrasp_filtered"):
                raise ValueError("--full-scene requires AnyGrasp or GraspNet")
            config = replace(config, full_scene=True, target_class=None)
        capture, targets = load_recording(args.input)
    backend = make_backend(config)
    try:
        if not synthetic and args.detect:
            # Ultralytics can change CUDA_VISIBLE_DEVICES when selecting CPU.
            if config.backend in ("anygrasp", "anygrasp_filtered", "graspnet"):
                backend.start()
            targets = make_detector(config).detect(capture.frame)
        result = plan_capture(capture, targets, backend, config)
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
    output = save_result(args.output, capture, result)
    if synthetic or not args.no_image:
        import cv2
        from rebot_grasp.visualization.overlay import draw_candidates
        if not cv2.imwrite(str(args.output / "preview.png"), draw_candidates(capture.frame, result.candidates)):
            raise RuntimeError("Could not save preview image")
    print(f"{result.status}: candidates={len(result.candidates)}, plans={len(result.plans)}")
    print(f"Saved {output.resolve()}; no hardware connected, no motion executed")
    if args.view_3d:
        from rebot_grasp.visualization.pointcloud_viewer import show_pointcloud
        show_pointcloud(
            capture.frame,
            result.candidates,
            min_depth_m=config.backend_options.get("min_depth_m", 0.05),
            max_depth_m=config.backend_options.get("max_depth_m", 1.0),
        )
    return 0 if result.candidates else 2
