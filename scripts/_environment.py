"""Configuration and software probes shared by check_env.sh and setup_env.sh."""
from importlib import import_module
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/default.yaml"


def project_package():
    import rebot_grasp
    if Path(rebot_grasp.__file__).resolve().parent != ROOT / "src/rebot_grasp":
        raise RuntimeError("rebot-grasp is installed from a different checkout")
    if not shutil.which("rebot-grasp"):
        raise RuntimeError("rebot-grasp CLI missing")


def settings(backend):
    from rebot_grasp.config import load_backend_options, load_profile
    from rebot_grasp.perception.detections import resolve_yolo_model_path

    cfg = load_profile(CONFIG)
    camera = cfg["camera"]["type"].lower()
    if camera.startswith("orbbec_"):
        camera_module = "pyorbbecsdk"
    elif camera.startswith("realsense_"):
        camera_module = "pyrealsense2"
    else:
        raise ValueError(f"Unsupported camera.type in {CONFIG}: {camera}")
    yolo = cfg["yolo"]
    name = yolo.get("model_name", "yoloe-26s-seg.pt")
    use_encoder = yolo.get("use_world", True) and yolo.get("custom_classes") and "yoloe-26" in name.lower()
    graspnet = {}
    anygrasp = {}
    filtered = {}
    if backend in ("all", "graspnet"):
        graspnet = {**load_backend_options(CONFIG, "graspnet"), **cfg.get("graspnet", {})}
    if backend in ("all", "anygrasp"):
        anygrasp = load_backend_options(CONFIG, "anygrasp")
        filtered = load_backend_options(CONFIG, "anygrasp_filtered")
    return dict(camera=camera, camera_module=camera_module,
                yolo=resolve_yolo_model_path(name, ROOT), encoder="yes" if use_encoder else "no",
                robot=(ROOT / Path(cfg["robot"]["repo_root"]).expanduser()).resolve(),
                baseline=graspnet.get("sdk_root", ""), checkpoint=graspnet.get("checkpoint", ""),
                anygrasp=anygrasp.get("sdk_root", ""), any_checkpoint=anygrasp.get("checkpoint", ""),
                any_filtered=filtered.get("sdk_root", ""), filtered_checkpoint=filtered.get("checkpoint", ""),
                calibration=CONFIG.parent / Path(cfg["calibration"]["hand_eye_path"]).expanduser())


def model_file(path):
    path = Path(path)
    with path.open("rb") as handle:
        prefix = handle.read(256).lstrip()
    if not prefix or prefix.startswith((b"<", b"version https://git-lfs")):
        raise ValueError(f"Expected a model file at {path}")
    if prefix.startswith(b"PK\x03\x04") and not zipfile.is_zipfile(path):
        raise ValueError(f"Incomplete model archive: {path}")


def graspnet_checkpoint(path):
    import torch
    model_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state.get("model_state_dict"):
        raise ValueError(f"Missing model_state_dict in {path}")


def opencv():
    import cv2
    from cv2 import aruco
    if not hasattr(aruco, "ArucoDetector") or not re.search(r"^\s*GUI:\s+(?!NONE\b)\S+", cv2.getBuildInformation(), re.M):
        raise RuntimeError("Install GUI-enabled opencv-python with ArUco support")


def cuda():
    import torch
    torch.zeros(1, device="cuda").item()
    return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"


def cuda_probe(component, sdk=None):
    import torch
    points = torch.rand(1, 32, 3, device="cuda")
    if component == "any_pointnet":
        from pointnet2.pointnet2_utils import furthest_point_sample
        shape = furthest_point_sample(points, 8).shape
        expected = (1, 8)
    elif component == "minkowski":
        import MinkowskiEngine as me
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
        sparse = me.SparseTensor(torch.ones(2, 3, device="cuda"), coordinates=coords, device="cuda")
        shape = me.MinkowskiConvolution(3, 4, kernel_size=1, dimension=3).cuda()(sparse).F.shape
        expected = (2, 4)
    elif component == "baseline_pointnet":
        sys.path.insert(0, str(Path(sdk) / "pointnet2"))
        from pointnet2._ext import cylinder_query, furthest_point_sampling
        shape = furthest_point_sampling(points, 8).shape
        expected = (1, 8)
    elif component == "knn":
        sys.path.insert(0, str(Path(sdk) / "knn"))
        from knn_modules import knn
        xyz = points.transpose(1, 2).contiguous()
        shape = knn(xyz, xyz, 1).shape
        expected = (1, 1, 32)
    else:
        raise ValueError(component)
    torch.cuda.synchronize()
    if shape != expected:
        raise RuntimeError(f"{component} returned {shape}, expected {expected}")


def robot_driver(repo):
    from rebot_grasp.robots.gripper_driver import ensure_rebot_sdk_in_syspath
    ensure_rebot_sdk_in_syspath(repo)
    from reBotArm_control_py.actuator import RebotArm


def camera_driver(name):
    if name == "pyorbbecsdk":
        from pyorbbecsdk import Config, OBAlignMode, OBFormat, OBSensorType, Pipeline
    elif name == "pyrealsense2":
        from pyrealsense2 import align, config, pipeline
    else:
        raise ValueError(name)


def gsnet(sdk):
    import torch
    from rebot_grasp.grasping.backends._anygrasp_sdk import load_sdk_module
    load_sdk_module(Path(sdk) / "grasp_detection")


def license_files(directory):
    config = directory / "licenseCfg.json"
    if not config.is_file() or not config.stat().st_size:
        raise FileNotFoundError(config)
    for suffix in ("*.lic", "*.public_key", "*.signature"):
        if not any(p.is_file() and p.stat().st_size for p in directory.glob(suffix)):
            raise FileNotFoundError(directory / suffix)


def baseline_model(sdk):
    from rebot_grasp.grasping.backends._graspnet_sdk import prepare_imports
    prepare_imports(Path(sdk))
    from graspnet import GraspNet


def probe(name, *args):
    if name in ("any_pointnet", "minkowski", "baseline_pointnet", "knn"):
        cuda_probe(name, *args)
    else:
        {"gsnet": gsnet, "baseline": baseline_model, "checkpoint": graspnet_checkpoint,
         "opencv": opencv, "project": project_package, "model": model_file,
         "camera": camera_driver}[name](*args)


def check(backend, robot):
    missing = []
    manual = []

    def item(label, fn, *args, user_action=False):
        try:
            detail = fn(*args)
        except Exception as exc:
            (manual if user_action else missing).append(label)
            print(f"[{'MANUAL' if user_action else 'MISSING'}] {label} — {exc}")
            return False
        print(f"[OK] {label}" + (f" — {detail}" if isinstance(detail, str) else ""))
        return True

    def native(name, *args):
        # The two SDKs load incompatible extensions with the same pointnet2 module name.
        # A separate interpreter also confines failures inside native libraries.
        result = subprocess.run([sys.executable, __file__, "probe", name, *args],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            lines = result.stdout.strip().splitlines()
            raise RuntimeError(lines[-1] if lines else f"probe exited {result.returncode}")

    item("Project package and CLI", project_package)
    for module in ("numpy", "yaml", "scipy", "torch", "torchvision", "ultralytics", "open3d"):
        item(module, import_module, module)
    item("OpenCV GUI and ArUco", opencv)
    use_gpu = backend != "obb"
    gpu = use_gpu and item("CUDA runtime", cuda)
    if use_gpu:
        item("graspnetAPI", import_module, "graspnetAPI")
    try:
        # Check repository configuration and assets even before editable installation.
        sys.path.insert(0, str(ROOT / "src"))
        cfg = settings(backend)
    except Exception as exc:
        print(f"[SKIP] Unable to resolve configured SDK and model paths: {exc}")
        return 1

    item(f"{cfg['camera']} SDK", camera_driver, cfg["camera_module"])
    item("YOLO segmentation weights", model_file, cfg["yolo"])
    if cfg["encoder"] == "yes":
        item("YOLOE text encoder", model_file, ROOT / "mobileclip2_b.ts")
        item("CLIP tokenizer", lambda: getattr(import_module("clip"), "tokenize"))
    if robot:
        from rebot_grasp.robots.gripper_driver import selected_hardware_yaml
        sdk_ok = item("B601 control SDK hardware configuration", lambda: selected_hardware_yaml(cfg["robot"]).name)
        item("Pinocchio", lambda: getattr(import_module("pinocchio"), "forwardKinematics"))
        motor_ok = item("MotorBridge", lambda: getattr(import_module("motorbridge"), "Controller"))
        for module in ("meshcat", "matplotlib"):
            item(module, import_module, module)
        if sdk_ok and motor_ok:
            item("B601 arm driver", robot_driver, cfg["robot"])
        from rebot_grasp.calibration.hand_eye import load_eye_in_hand
        item("Hand–eye matrix (see examples/hand_eye/README.md)", load_eye_in_hand,
             cfg["calibration"], user_action=True)

    if backend in ("all", "anygrasp"):
        for sdk in dict.fromkeys((cfg["anygrasp"], cfg["any_filtered"])):
            item(f"AnyGrasp gsnet binary ({sdk})", native, "gsnet", sdk)
            item("AnyGrasp machine license files (see docs/sdk_setup.md)", license_files,
                 Path(sdk) / "grasp_detection/license", user_action=True)
        for path in dict.fromkeys((cfg["any_checkpoint"], cfg["filtered_checkpoint"])):
            item("AnyGrasp detection checkpoint (provided with authorization)", model_file,
                 path, user_action=True)
        if gpu:
            item("AnyGrasp MinkowskiEngine CUDA operator", native, "minkowski")
            item("AnyGrasp PointNet2 CUDA operator", native, "any_pointnet")
    if backend in ("all", "graspnet"):
        sdk = cfg["baseline"]
        item("GraspNet-Baseline SDK and model import", native, "baseline", sdk)
        item("GraspNet checkpoint", graspnet_checkpoint, cfg["checkpoint"])
        if gpu:
            item("GraspNet PointNet2 CUDA operator", native, "baseline_pointnet", sdk)
            item("GraspNet kNN CUDA operator", native, "knn", sdk)
    if use_gpu and not gpu:
        print("[SKIP] CUDA operator tests require a working NVIDIA driver and CUDA PyTorch.")
    if robot:
        print("Before hardware use, confirm the selected RS/DM variant, CAN channel and camera USB permissions.")
        collector = ROOT.parent / "rebot-arm-b601-robstride"
        if not (collector / "scripts/wiki_calibration.py").is_file():
            print("[OPTIONAL] Stationary calibration needs the external collector described in examples/stationary_calibration/README.md.")
    print(f"Software checks failed: {len(missing)}. Outstanding manual items: {len(manual)}.")
    if missing:
        return 1
    return 3 if manual else 0


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    try:
        if command == "check":
            sys.exit(check(arguments[0], arguments[1] == "1"))
        elif command == "settings":
            for key, value in settings(arguments[0]).items():
                print(f"{key}={value}")
        elif command == "probe":
            probe(*arguments)
        else:
            raise ValueError(command)
    except Exception as error:
        sys.exit(f"{type(error).__name__}: {error}")
