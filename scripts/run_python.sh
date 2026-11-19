#!/usr/bin/env bash
# Anchor config/model paths to this checkout. REBOT_PYTHON selects a custom environment.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
python_bin=${REBOT_PYTHON:-python}
paths="$root/src:$root/sdk/reBotArm_control_py"
conda_exe=${CONDA_EXE:-$HOME/miniconda3/bin/conda}
if [[ -z ${REBOT_PYTHON:-} && ${CONDA_DEFAULT_ENV:-} != rebot && -x "$conda_exe" ]]; then
    conda_base=$(cd "$(dirname "$conda_exe")/.." && pwd)
    if [[ ${CONDA_DEFAULT_ENV:-} == LJ || -x "$conda_base/envs/LJ/bin/python" ]]; then
        unset PYTHONPATH LD_LIBRARY_PATH
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate LJ
        unset PYTHONPATH LD_LIBRARY_PATH
        export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
        if [[ -x "$CONDA_PREFIX/bin/nvcc" ]]; then export CUDA_HOME="$CONDA_PREFIX"; fi
        paths="$root/sdk/runtime/gui:$root/sdk/runtime/python:$paths"
    fi
fi
export PYTHONPATH="$paths${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -u "$@"
