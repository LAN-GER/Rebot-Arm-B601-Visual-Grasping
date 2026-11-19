#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo 'Usage: bash scripts/setup_graspnet_env.sh [--robot]'
    echo 'Create or complete the rebot Conda environment for GraspNet camera preview.'
    echo 'Use --robot to also install B601 software dependencies.'
    echo 'GRASPNET_CHECKPOINT_URL can point to a trusted alternate checkpoint download.'
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --robot|"") ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi
mode=${1:-}

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
conda_exe=${CONDA_EXE:-conda}
if ! command -v "$conda_exe" >/dev/null; then
    if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        conda_exe="$HOME/miniconda3/bin/conda"
    else
        echo 'Install Conda first to create the rebot environment.' >&2
        exit 1
    fi
fi
for command in git curl nvidia-smi; do
    command -v "$command" >/dev/null || { echo "Required system command: $command" >&2; exit 1; }
done
if ! nvidia-smi -L; then
    echo 'GraspNet requires an accessible NVIDIA GPU and a working NVIDIA driver.' >&2
    exit 1
fi

conda_base=$("$conda_exe" info --base)
registered_envs=$("$conda_exe" env list --json)
env_exists=$("$conda_base/bin/python" -c '
import json, sys
from pathlib import Path
print("yes" if any(Path(p).name == "rebot" for p in json.load(sys.stdin)["envs"]) else "no")
' <<<"$registered_envs")
if [[ "$env_exists" == no ]]; then
    "$conda_exe" create -y -n rebot python=3.10 pip
fi
# Conda and compiler activation hooks can refer to unset shell variables.
set +u
source "$conda_base/etc/profile.d/conda.sh"
conda activate rebot
set -u
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 in the rebot environment"'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# Retain an existing CUDA-enabled PyTorch build. Replace a CPU-only build with
# the CUDA 12.1 wheel set used by the project's SDK guide.
torch_cuda=$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)
if [[ -z "$torch_cuda" ]]; then
    python -m pip install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121' \
        --index-url https://download.pytorch.org/whl/cu121
    torch_cuda=12.1
fi
python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; torch.zeros(1, device="cuda").item()'

# pip keeps installed versions that already satisfy these requirements.
python -m pip install -e '.[obb,yolo,visualization]' scipy wheel ninja graspnetAPI
config_values=$(python - <<'PY'
from pathlib import Path
from rebot_grasp.config import load_example_config
from rebot_grasp.perception.detections import resolve_yolo_model_path

root = Path.cwd()
cfg = load_example_config(root / "config/default.yaml")
options = cfg["graspnet"]
yolo = cfg["yolo"]
model_name = yolo.get("model_name", "yoloe-26s-seg.pt")
print(cfg["camera"]["type"])
print(options["sdk_root"])
print(options["checkpoint"])
print(resolve_yolo_model_path(model_name, root))
print("yes" if yolo.get("use_world", True) and yolo.get("custom_classes") and "yoloe-26" in model_name.lower() else "no")
print((root / Path(cfg["robot"]["repo_root"]).expanduser()).resolve())
PY
)
mapfile -t settings <<<"$config_values"
camera_type=${settings[0]}
sdk_root=${settings[1]}
checkpoint=${settings[2]}
yolo_weights=${settings[3]}
use_text_encoder=${settings[4]}
robot_sdk=${settings[5]}
case "$camera_type" in
    orbbec_*) python -c 'import pyorbbecsdk' || python -m pip install 'pyorbbecsdk2==2.0.18' ;;
    realsense_*) python -c 'import pyrealsense2' || python -m pip install -e '.[realsense]' ;;
    *) echo "Unsupported camera in config/default.yaml: $camera_type" >&2; exit 1 ;;
esac

if [[ ! -e "$sdk_root" ]]; then
    mkdir -p "$(dirname "$sdk_root")"
    git clone https://github.com/graspnet/graspnet-baseline.git "$sdk_root"
fi
if [[ ! -f "$sdk_root/models/graspnet.py" ]]; then
    echo "GraspNet-Baseline source is missing at $sdk_root. Check sdk_root in config/backends/graspnet.yaml." >&2
    exit 1
fi

needs_pointnet2=0
needs_knn=0
PYTHONPATH="$sdk_root/pointnet2" python -c '
import torch
from pointnet2._ext import cylinder_query, furthest_point_sampling
x = torch.rand(1, 32, 3, device="cuda")
assert furthest_point_sampling(x, 8).shape == (1, 8)
torch.cuda.synchronize()
' 2>/dev/null || needs_pointnet2=1
PYTHONPATH="$sdk_root/knn" python -c '
import torch
from knn_modules import knn
x = torch.zeros((1, 3, 8), device="cuda")
knn(x, x, 1)
torch.cuda.synchronize()
' 2>/dev/null || needs_knn=1

if (( needs_pointnet2 || needs_knn )); then
    # Locate a toolkit for the installed PyTorch CUDA version.
    nvcc=$(command -v nvcc || true)
    nvcc_root=""
    if [[ -n "$nvcc" ]]; then
        nvcc_root=$(dirname "$(dirname "$(readlink -f "$nvcc")")")
    fi
    cuda_root=""
    for candidate in "${CUDA_HOME:-}" "$CONDA_PREFIX" "/usr/local/cuda-$torch_cuda" "$nvcc_root"; do
        if [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] &&
            "$candidate/bin/nvcc" --version | grep -F "release $torch_cuda," >/dev/null; then
            cuda_root=$candidate
            break
        fi
    done
    if [[ -z "$cuda_root" ]]; then
        conda install -y -c "nvidia/label/cuda-${torch_cuda}.0" "cuda-toolkit=$torch_cuda"
        cuda_root=$CONDA_PREFIX
    fi
    if [[ ! -x "$cuda_root/bin/nvcc" ]] ||
        ! "$cuda_root/bin/nvcc" --version | grep -F "release $torch_cuda," >/dev/null; then
        echo "Could not locate nvcc for CUDA $torch_cuda." >&2
        exit 1
    fi
    export CUDA_HOME="$cuda_root"
    export PATH="$CUDA_HOME/bin:$PATH"
    export MAX_JOBS=${MAX_JOBS:-2}
    host_cxx=${CXX:-c++}
    host_compiler_version=$("$host_cxx" -dumpversion 2>/dev/null || true)
    if [[ -z "$host_compiler_version" ]] ||
        [[ "$torch_cuda" == 12.1 && "${host_compiler_version%%.*}" -gt 12 ]]; then
        conda install -y -c conda-forge 'gcc_linux-64=12' 'gxx_linux-64=12'
        set +u
        conda activate rebot
        set -u
    fi
    if (( needs_pointnet2 )); then
        python -m pip install --no-build-isolation --no-deps --upgrade \
            --target "$sdk_root/pointnet2" "$sdk_root/pointnet2"
        # Prefer the SDK-local extension if another backend provides pointnet2.
        touch "$sdk_root/pointnet2/pointnet2/__init__.py"
    fi
    if (( needs_knn )); then
        python -m pip install --no-build-isolation --no-deps --upgrade \
            --target "$sdk_root/knn" "$sdk_root/knn"
    fi
fi

partial_download=""
trap 'if [[ -n "$partial_download" ]]; then rm -f -- "$partial_download"; fi' EXIT
download() {
    local destination=$1
    shift
    mkdir -p "$(dirname "$destination")"
    partial_download=$(mktemp "${destination}.part.XXXXXX")
    "$@" "$partial_download" || return
    if [[ ! -s "$partial_download" ]]; then
        echo "Download is empty: $destination" >&2
        return 1
    fi
}
finish_download() {
    mv -- "$partial_download" "$1"
    partial_download=""
}

if [[ ! -s "$yolo_weights" ]]; then
    if [[ "$(basename "$yolo_weights")" != yoloe-26s-seg.pt ]]; then
        echo "Place your configured YOLO model at $yolo_weights." >&2
        exit 1
    fi
    download "$yolo_weights" curl -fLsS --retry 3 \
        https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt -o
    finish_download "$yolo_weights"
fi
if [[ "$use_text_encoder" == yes ]]; then
    python -c 'import clip; assert hasattr(clip, "tokenize")' ||
        python -m pip install 'git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40'
    encoder="$root/mobileclip2_b.ts"
    if [[ ! -s "$encoder" ]]; then
        download "$encoder" curl -fLsS --retry 3 \
            https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts -o
        finish_download "$encoder"
    fi
fi

if [[ "$mode" == --robot ]]; then
    if [[ ! -e "$robot_sdk" ]]; then
        mkdir -p "$(dirname "$robot_sdk")"
        git clone https://github.com/Seeed-Projects/reBotArm_control_py.git "$robot_sdk"
    fi
    python -c 'import pinocchio' || conda install -y -c conda-forge 'pinocchio=3.9.0' 'numpy<2'
    python -m pip install 'motorbridge>=0.6.0' 'meshcat>=0.3.2' 'matplotlib>=3.10'
    PYTHONPATH="$robot_sdk" python -c 'import motorbridge, pinocchio; from reBotArm_control_py.actuator import RebotArm'
fi

if [[ ! -s "$checkpoint" ]]; then
    checkpoint_url=${GRASPNET_CHECKPOINT_URL:-}
    if [[ -z "$checkpoint_url" ]]; then
        case "$(basename "$checkpoint")" in
            checkpoint-rs.tar) drive_id=1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk ;;
            checkpoint-kn.tar) drive_id=1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX ;;
            *) echo "Place your custom GraspNet checkpoint at $checkpoint." >&2; exit 1 ;;
        esac
        checkpoint_url="https://drive.usercontent.google.com/download?id=$drive_id&export=download&confirm=t"
    fi
    if download "$checkpoint" curl -fLsS --retry 3 "$checkpoint_url" -o &&
        python - "$partial_download" <<'PY'
from pathlib import Path
import sys

import torch

path = Path(sys.argv[1])
with path.open("rb") as file:
    if file.read(256).lstrip().startswith(b"<"):
        raise SystemExit("The download returned an HTML page. Check for a Google Drive quota or access error.")
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
    raise SystemExit("The downloaded file is not a GraspNet checkpoint")
PY
    then
        finish_download "$checkpoint"
    else
        printf 'Download the checkpoint from one of the official links and place it at %s\n' "$checkpoint" >&2
        echo 'https://github.com/graspnet/graspnet-baseline#training-and-testing' >&2
        exit 1
    fi
fi

export CONDA_EXE="$conda_exe"
bash "$root/scripts/check_graspnet_env.sh"
if [[ "$mode" == --robot ]]; then
    printf '%s\n' 'Run the checker with --robot after configuring the hardware and completing hand–eye calibration.'
fi
