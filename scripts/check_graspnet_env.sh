#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/check_graspnet_env.sh [--robot]'
    echo 'Check GraspNet camera-preview dependencies in the rebot environment.'
    echo 'Use --robot to also check robot SDK imports and the hand–eye file.'
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --robot|"") ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
conda_exe=${CONDA_EXE:-conda}
if ! command -v "$conda_exe" >/dev/null; then
    if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        conda_exe="$HOME/miniconda3/bin/conda"
    else
        echo '[MISSING] Conda. Install Conda to create the rebot environment.' >&2
        exit 1
    fi
fi

if command -v nvidia-smi >/dev/null && nvidia-smi -L; then
    echo '[OK] NVIDIA driver'
else
    echo '[MISSING OR UNUSABLE] NVIDIA driver or GPU access' >&2
fi

if ! "$conda_exe" run -n rebot python -c 'import sys; print("[OK] rebot Python", sys.version.split()[0])'; then
    echo '[MISSING] The rebot environment or its Python interpreter. Run scripts/setup_graspnet_env.sh.' >&2
    exit 1
fi

export PYTHONPATH="$root/src:$root/sdk/reBotArm_control_py"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$conda_exe" run -n rebot --no-capture-output python - "${1:-}" <<'PY'
import importlib
from pathlib import Path
import re
import sys

issues = []


def check(label, verify):
    try:
        verify()
    except Exception as error:
        issues.append(label)
        print(f"[MISSING OR UNUSABLE] {label}: {error}")
    else:
        print(f"[OK] {label}")


def require_file(path):
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)


def opencv_gui():
    import cv2
    if not re.search(r"^\s*GUI:\s*(?!NONE\b)\S+", cv2.getBuildInformation(), re.M):
        raise RuntimeError("OpenCV needs GUI support for the preview window")


def cuda_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU unavailable to PyTorch {torch.__version__} (CUDA build {torch.version.cuda})")
    torch.ones(2, device="cuda").sum().item()
    print(f"     {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")


for module in ("numpy", "yaml", "scipy", "open3d", "ultralytics", "torchvision", "graspnetAPI"):
    check(module, lambda name=module: importlib.import_module(name))
check("OpenCV GUI", opencv_gui)
check("CUDA runtime", cuda_device)


def assets_and_sdk():
    from rebot_grasp.config import load_example_config
    from rebot_grasp.perception.detections import resolve_yolo_model_path
    from rebot_grasp.grasping.backends._graspnet_sdk import prepare_imports

    root = Path.cwd()
    cfg = load_example_config(root / "config/default.yaml")
    camera_type = cfg["camera"]["type"]
    camera_module = "pyorbbecsdk" if camera_type.startswith("orbbec_") else "pyrealsense2"
    check(f"{camera_type} driver", lambda: importlib.import_module(camera_module))
    yolo = cfg["yolo"]
    model_name = yolo.get("model_name", "yoloe-26s-seg.pt")
    weights = resolve_yolo_model_path(model_name, root)
    check("YOLO weights", lambda: require_file(weights))
    if yolo.get("use_world", True) and yolo.get("custom_classes") and "yoloe-26" in model_name.lower():
        check("YOLOE text encoder", lambda: require_file(root / "mobileclip2_b.ts"))
        def clip_tokenizer():
            from clip import tokenize

        check("Ultralytics CLIP", clip_tokenizer)

    options = cfg["graspnet"]
    sdk = Path(options["sdk_root"])
    check("GraspNet checkpoint", lambda: require_file(Path(options["checkpoint"])))

    def operators():
        import torch
        prepare_imports(sdk)
        from graspnet import GraspNet
        from pointnet2_utils import furthest_point_sample
        from knn_modules import knn
        points = torch.rand(1, 32, 3, device="cuda")
        xyz = points.transpose(1, 2).contiguous()
        if furthest_point_sample(points, 8).shape != (1, 8) or knn(xyz, xyz, 1).shape != (1, 1, 32):
            raise RuntimeError("unexpected CUDA operator output shape")
        torch.cuda.synchronize()

    check("GraspNet SDK with PointNet2 and kNN CUDA operators", operators)

    if sys.argv[1] == "--robot":
        from rebot_grasp.calibration.hand_eye import load_eye_in_hand
        from rebot_grasp.robots.gripper_driver import ensure_rebot_sdk_in_syspath

        def robot_sdk():
            ensure_rebot_sdk_in_syspath(cfg["robot"].get("repo_root"))
            from reBotArm_control_py.actuator import RebotArm
            import pinocchio
            import motorbridge

        check("B601 control SDK", robot_sdk)
        calibration = root / "config/calibration" / camera_type / "hand_eye.npz"
        check("Eye-in-hand calibration", lambda: load_eye_in_hand(calibration))


if "numpy" not in issues and "yaml" not in issues:
    check("GraspNet project configuration", assets_and_sdk)

if issues:
    print(f"Checks failed: {len(issues)}. Run scripts/setup_graspnet_env.sh to fill software and asset gaps.")
    if sys.argv[1] == "--robot":
        print("Generate missing hand–eye calibration separately with examples/hand_eye/run.sh.")
    raise SystemExit(1)
print("GraspNet environment checks passed.")
PY
