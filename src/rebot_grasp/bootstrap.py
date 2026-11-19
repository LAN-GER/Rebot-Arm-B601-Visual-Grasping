"""The only place that selects concrete implementations from configuration."""

from rebot_grasp.config import AppConfig


def make_backend(config: AppConfig):
    options = dict(config.backend_options)
    if config.backend == "obb":
        from rebot_grasp.grasping.backends.obb import OBBBackend
        return OBBBackend(**options)
    if config.backend == "obb_filtered":
        from rebot_grasp.grasping.backends.obb_filtered import FilteredOBBBackend
        return FilteredOBBBackend(**options)
    if config.backend in ("anygrasp", "anygrasp_filtered"):
        from rebot_grasp.grasping.backends.anygrasp import AnyGraspBackend
        from rebot_grasp.grasping.backends._anygrasp_sdk import SDKDetector
        detector = SDKDetector(options.pop("sdk_root"), options.pop("checkpoint"),
                               options.pop("max_gripper_width_m"), options.pop("gripper_height_m"),
                               options.pop("timeout_s", 120))
        if config.backend == "anygrasp_filtered":
            from rebot_grasp.grasping.backends.anygrasp_filtered import FilteredAnyGraspBackend
            return FilteredAnyGraspBackend(detector, config.filters, **options)
        return AnyGraspBackend(detector, **options)
    if config.backend == "graspnet":
        from rebot_grasp.grasping.backends.graspnet import GraspNetBackend
        return GraspNetBackend(**options)
    raise ValueError(f"Unknown backend: {config.backend}")


def make_detector(config: AppConfig):
    from rebot_grasp.perception.yolo import YoloDetector
    return YoloDetector(**config.yolo)


def make_camera(config: AppConfig):
    camera = config.camera
    kind = camera["type"]
    dimensions = {
        "width": camera.get("color_width", 1280),
        "height": camera.get("color_height", 720),
        "fps": camera.get("fps", 30),
    }
    if kind in {"orbbec_gemini2", "orbbec_gemini336"}:
        from rebot_grasp.cameras.orbbec import OrbbecCamera
        return OrbbecCamera(**dimensions)
    if kind in {"realsense_d405", "realsense_d435i"}:
        from rebot_grasp.cameras.realsense import RealSenseCamera
        return RealSenseCamera(**dimensions, serial=camera.get("serial"))
    raise ValueError(f"Unsupported camera type: {kind}")


def make_robot(config: AppConfig):
    from rebot_grasp.robots.b601 import B601Session
    return B601Session(config.robot, verify_open=config.backend in ("anygrasp", "anygrasp_filtered"))


def execution_calibration(config: AppConfig, config_path):
    from pathlib import Path

    from rebot_grasp.calibration.hand_eye import load_eye_in_hand

    if config.planning["T_grasp_tcp"] is None:
        raise ValueError("Set planning.T_grasp_tcp before execution")
    hand_eye_path = config.calibration.get("hand_eye_path")
    if not hand_eye_path:
        raise ValueError("Set calibration.hand_eye_path before execution")
    return load_eye_in_hand(config_path.parent / Path(hand_eye_path).expanduser())
