#!/usr/bin/env bash
# Install the software used by the selected project examples, then run the same checker.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/scripts/_environment.sh"
action=setup
env_options "$@"
cd "$root"
REBOT_INSTALL_JOURNAL=$(mktemp)
export REBOT_INSTALL_JOURNAL
install_summary() {
    printf '\n----\nInstallation summary / 本次安装变更\n'
    if [[ -s "$REBOT_INSTALL_JOURNAL" ]]; then
        cat "$REBOT_INSTALL_JOURNAL"
    else
        echo 'No packages or SDKs changed. / 本次无变更。'
    fi
    rm -f -- "$REBOT_INSTALL_JOURNAL"
}
trap install_summary EXIT
if [[ "$(uname -s)/$(uname -m)" != Linux/x86_64 ]]; then
    echo 'This installer supports Ubuntu x86-64. See docs/sdk_setup.md for manual setup.' >&2
    exit 1
fi
for tool in git curl; do
    command -v "$tool" >/dev/null || { echo "Install $tool, then rerun." >&2; exit 1; }
done
stage='Preparing the rebot Conda environment'
trap 'echo "Setup stopped during: $stage. See the error above and rerun." >&2' ERR
install_miniforge() (
    local prefix="$HOME/miniforge3" installer
    if [[ -e "$prefix" ]]; then
        echo "Conda was not found in $prefix. Repair that installation or set CONDA_EXE to a working conda executable." >&2
        return 1
    fi
    installer=$(mktemp)
    trap 'rm -f -- "$installer"' EXIT
    echo "Installing Miniforge in $prefix"
    curl -fLsS --retry 3 --connect-timeout 20 -o "$installer" \
        https://github.com/conda-forge/miniforge/releases/download/26.9.4-2/Miniforge3-26.9.4-2-Linux-x86_64.sh
    printf '%s  %s\n' bbaf16b3897da24c4be50315934ba74848f617b4495982841d6a0a60e0edbbdf "$installer" | sha256sum --check
    bash "$installer" -b -p "$prefix"
    echo "[installer] Installed Miniforge 26.9.4-2 ($prefix)" >> "$REBOT_INSTALL_JOURNAL"
    printf 'To activate Conda in your shell: source "%s/etc/profile.d/conda.sh"\n' "$prefix"
)
if ! conda_exe=$(find_conda); then
    install_miniforge
    conda_exe="$HOME/miniforge3/bin/conda"
fi
conda_base=$("$conda_exe" info --base)
package_helper="$root/scripts/_install_packages.py"
conda_install() { "$conda_base/bin/python" "$package_helper" conda "$@"; }
pip_install() { python "$package_helper" pip "$@"; }
satisfies() { python "$package_helper" satisfies "$@"; }
created=0
env_dirs=$("$conda_exe" env list --json)
if ! "$conda_base/bin/python" -c '
import json, sys
from pathlib import Path
sys.exit(not any(Path(p).name == "rebot" for p in json.load(sys.stdin)["envs"]))
' <<<"$env_dirs"; then
    echo 'Creating rebot with Python 3.10'
    "$conda_exe" create -y -n rebot --override-channels -c conda-forge python=3.10 pip
    created=1
fi
set +u
source "$conda_base/etc/profile.d/conda.sh"
conda activate rebot
set -u
if (( created )); then
    "$conda_base/bin/python" "$package_helper" created "$CONDA_PREFIX"
fi
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$CONDA_PREFIX/bin/python" ]] || ! python -c 'import sys; sys.exit(sys.version_info[:2] != (3,10))'; then
    conda_install -c conda-forge python=3.10 pip
    hash -r
fi
python -c 'import sys; print(sys.executable, sys.version.split()[0])'
if ! satisfies 'pip>=22.2' >/dev/null 2>&1; then
    conda_install -c conda-forge 'pip>=22.2' 'python=3.10'
fi

probe() { python "$root/scripts/_environment.py" probe "$@"; }
works() { probe "$@" >/dev/null 2>&1; }
has() { python -c "$1" >/dev/null 2>&1; }

stage='Preparing PyTorch, the project and camera driver'
echo "$stage"
gpu=0
if [[ "$backend" != obb ]] && command -v nvidia-smi >/dev/null && nvidia-smi -L; then
    gpu=1
fi
if (( gpu )); then
    torch_cuda=$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)
    if [[ -z "$torch_cuda" ]]; then
        pip_install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121' \
            --index-url https://download.pytorch.org/whl/cu121
        torch_cuda=12.1
    fi
    if ! has 'import torchvision; from torchvision.ops import nms'; then
        torch_version=$(python -c 'import torch; print(torch.__version__)')
        pip_install "torch==$torch_version" torchvision --index-url "https://download.pytorch.org/whl/cu${torch_cuda/.}"
    fi
    if ! has 'import torch; torch.ones(1, device="cuda").item()'; then
        echo "Check the driver and install PyTorch wheels that support your GPU. See docs/sdk_setup.md." >&2
        gpu=0
    fi
elif ! has 'import torch, torchvision'; then
    pip_install -r requirements-pytorch.txt --index-url https://download.pytorch.org/whl/cpu
fi
if ! works project || ! works opencv || ! satisfies 'numpy>=1.23,<2' 'opencv-python>=4.7,<4.12' 'open3d>=0.17' 'PyYAML>=6' || ! has '
import numpy, scipy, open3d, ultralytics
assert numpy.__version__.startswith("1.") and ultralytics.__version__ == "8.4.36"
'; then
    pip_install -e '.[obb,yolo,visualization]' scipy
    if ! works opencv; then
        pip_install --force-reinstall --no-deps 'opencv-python>=4.7,<4.12'
    fi
fi

config_values=$(python scripts/_environment.py settings "$backend")
declare -A cfg
while IFS='=' read -r key value; do cfg["$key"]=$value; done <<<"$config_values"
camera_module=${cfg[camera_module]}
yolo_weights=${cfg[yolo]}
use_encoder=${cfg[encoder]}
robot_sdk=${cfg[robot]}
baseline=${cfg[baseline]}
checkpoint=${cfg[checkpoint]}
any_sdk=${cfg[anygrasp]}
case "$camera_module" in
    pyorbbecsdk) { works camera "$camera_module" && satisfies 'pyorbbecsdk2==2.0.18'; } || pip_install 'pyorbbecsdk2==2.0.18' ;;
    pyrealsense2) { works camera "$camera_module" && satisfies 'pyrealsense2>=2.54'; } || pip_install 'pyrealsense2>=2.54' ;;
esac

source "$root/scripts/_sdk_assets.sh"

install_yolo() {
    if ! works model "$yolo_weights"; then
        if [[ "$(basename "$yolo_weights")" != yoloe-26s-seg.pt ]]; then
            echo "Place your configured segmentation model at $yolo_weights." >&2
            return 1
        fi
        install_model "$yolo_weights" https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt
    fi
    if [[ "$use_encoder" == yes ]]; then
        has 'from clip import tokenize' || pip_install 'git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40'
        install_model "$root/mobileclip2_b.ts" https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts
    fi
}

install_robot() {
    clone_sdk "$robot_sdk" reBotArm_control_py/actuator/rebotarm.py https://github.com/Seeed-Projects/reBotArm_control_py.git
    if ! has 'import pinocchio; from pinocchio import forwardKinematics; assert pinocchio.__version__ == "3.9.0"'; then
        conda_install -c conda-forge 'pinocchio=3.9.0' 'numpy<2'
    fi
    if ! has 'from motorbridge import Controller, Mode, CallError; import meshcat, matplotlib' ||
        ! satisfies 'motorbridge==0.6.0' 'meshcat>=0.3.2' 'matplotlib>=3.10'; then
        pip_install 'motorbridge==0.6.0' 'meshcat>=0.3.2' 'matplotlib>=3.10'
    fi
}

prepare_toolkit() {
    local candidate nvcc nvcc_dir="" version label cuda_root=""
    nvcc=$(command -v nvcc || true)
    if [[ -n "$nvcc" ]]; then nvcc_dir=$(dirname "$(dirname "$(readlink -f "$nvcc")")"); fi
    for candidate in "${CUDA_HOME:-}" "$CONDA_PREFIX" "/usr/local/cuda-$torch_cuda" "$nvcc_dir"; do
        if [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] &&
            "$candidate/bin/nvcc" --version | grep -F "release $torch_cuda," >/dev/null; then
            cuda_root=$candidate
            break
        fi
    done
    if [[ -z "$cuda_root" ]]; then
        label="cuda-${torch_cuda}.0"
        if [[ "$torch_cuda" == 12.1 ]]; then label=cuda-12.1.1; fi
        conda_install -c "nvidia/label/$label" -c conda-forge "cuda-toolkit=$torch_cuda"
        cuda_root=$CONDA_PREFIX
        "$cuda_root/bin/nvcc" --version | grep -F "release $torch_cuda,"
    fi
    export CUDA_HOME="$cuda_root"
    version=$("${CXX:-c++}" -dumpversion 2>/dev/null || true)
    if [[ -z "$version" || ( "$torch_cuda" == 12.1 && "${version%%.*}" -gt 12 ) ]]; then
        conda_install -c conda-forge 'gcc_linux-64=12' 'gxx_linux-64=12'
        set +u
        conda activate rebot
        set -u
    fi
    export PATH="$CUDA_HOME/bin:$PATH"
    export MAX_JOBS=${MAX_JOBS:-2}
    pip_install wheel ninja
}

install_anygrasp() {
    clone_sdk "$any_sdk" pointnet2/setup.py https://github.com/graspnet/anygrasp_sdk.git
    if [[ "${cfg[any_filtered]}" != "$any_sdk" ]]; then
        clone_sdk "${cfg[any_filtered]}" pointnet2/setup.py https://github.com/graspnet/anygrasp_sdk.git
    fi
    if (( ! gpu )); then echo 'AnyGrasp CUDA builds deferred until a working GPU is available.'; return; fi
    local need_point=0 need_me=0 branch
    works any_pointnet || need_point=1
    works minkowski || need_me=1
    if (( need_point || need_me )); then prepare_toolkit; fi
    if (( need_point )); then
        pip_install --no-build-isolation --no-deps --force-reinstall "$any_sdk/pointnet2"
    fi
    if (( need_me )); then
        case "$torch_cuda" in
            11.*) branch=master ;; 12.*) branch=cuda-12-1 ;; 13.*) branch=cuda-13 ;;
            *) echo "Check the AnyGrasp SDK guide for CUDA $torch_cuda." >&2; return 1 ;;
        esac
        local source="$root/sdk/MinkowskiEngine-$branch"
        clone_sdk "$source" setup.py https://github.com/chenxi-wang/MinkowskiEngine.git --branch "$branch"
        conda_install -c conda-forge 'openblas>=0.3.30,<0.4' 'numpy<2'
        python -c 'import ctypes, sys; ctypes.CDLL(sys.prefix + "/lib/libopenblas.so")'
        # Repairing the BLAS runtime can also restore an already-built extension.
        if works minkowski; then return; fi
        (
            cd "$source"
            local wheels
            wheels=$(mktemp -d)
            trap 'rm -rf -- "$wheels"' EXIT
            # Upstream setup.py invokes pip uninstall while building. Keep that
            # subprocess inside a disposable venv until the wheel is reviewed.
            python -m venv --system-site-packages "$wheels/build-env"
            PATH="$wheels/build-env/bin:$PATH" "$wheels/build-env/bin/python" setup.py bdist_wheel --dist-dir "$wheels" --force_cuda --blas=openblas \
                --blas_include_dirs="$CONDA_PREFIX/include" --blas_library_dirs="$CONDA_PREFIX/lib"
            pip_install --no-deps --force-reinstall "$wheels"/*.whl
        )
    fi
}

install_graspnet() {
    clone_sdk "$baseline" models/graspnet.py https://github.com/graspnet/graspnet-baseline.git
    if (( ! gpu )); then echo 'GraspNet CUDA builds deferred until a working GPU is available.'; return; fi
    local need_point=0 need_knn=0
    works baseline_pointnet "$baseline" || need_point=1
    works knn "$baseline" || need_knn=1
    if (( need_point || need_knn )); then prepare_toolkit; fi
    if (( need_point )); then
        pip_install --no-build-isolation --no-deps --upgrade --target "$baseline/pointnet2" "$baseline/pointnet2"
        touch "$baseline/pointnet2/pointnet2/__init__.py"
    fi
    if (( need_knn )); then
        pip_install --no-build-isolation --no-deps --upgrade --target "$baseline/knn" "$baseline/knn"
    fi
}

install_graspnet_weights() {
    if works checkpoint "$checkpoint"; then return; fi
    local url=${GRASPNET_CHECKPOINT_URL:-} id
    if [[ -z "$url" && -z "$checkpoint_file" ]]; then
        case "$(basename "$checkpoint")" in
            checkpoint-rs.tar) id=1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk ;;
            checkpoint-kn.tar) id=1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX ;;
            *) echo "Supply --checkpoint-file PATH for $checkpoint." >&2; return 1 ;;
        esac
        url="https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
    fi
    echo 'If Google Drive limits the download, rerun with --checkpoint-file PATH.'
    install_model "$checkpoint" "$url" checkpoint "$checkpoint_file"
}

install_graspnet_api() {
    { has 'import graspnetAPI, transforms3d' && satisfies 'transforms3d>=0.4.2'; } || pip_install 'transforms3d>=0.4.2' \
        'graspnetAPI @ git+https://github.com/graspnet/graspnetAPI.git@eb57dd2092d8dbe05312a29c3d0c22f3226efbfc'
}

# Finish independent SDKs and print a combined report even if one download/build fails.
failed=()
run_component() {
    stage=$1
    echo "== $stage =="
    set +e
    (set -e; "$2")
    local status=$?
    set -e
    if (( status )); then failed+=("$stage"); fi
}
trap - ERR
run_component 'YOLOE model and text encoder' install_yolo
if (( robot )); then run_component 'B601 control SDK' install_robot; fi
if [[ "$backend" != obb ]]; then run_component 'graspnetAPI' install_graspnet_api; fi
if [[ "$backend" == all || "$backend" == anygrasp ]]; then run_component 'AnyGrasp SDK' install_anygrasp; fi
if [[ "$backend" == all || "$backend" == graspnet ]]; then
    run_component 'GraspNet-Baseline SDK' install_graspnet
    run_component 'GraspNet checkpoint' install_graspnet_weights
fi
options=(--backend "$backend")
if (( robot )); then options+=(--robot); fi
status=0
bash "$root/scripts/check_env.sh" "${options[@]}" || status=$?
if (( ${#failed[@]} )); then
    printf 'Setup step failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
if (( status == 0 )); then echo 'Ready. Run: conda activate rebot'; fi
exit "$status"
