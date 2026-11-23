#!/usr/bin/env bash
# Manage the machine-bound license and detection weights for the configured AnyGrasp backend.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/scripts/env/_environment.sh"

fail() { printf '\n----\n[ERROR] %s\n' "$1" >&2; exit 1; }
usage='Usage: bash scripts/anygrasp_license.sh [--backend anygrasp|anygrasp_filtered] [1 | 2 [DELIVERY_PATH]]'
if [[ ${1:-} == -h || ${1:-} == --help ]]; then echo "$usage"; exit 0; fi
backend=anygrasp
if [[ ${1:-} == --backend ]]; then
    case "${2:-}" in anygrasp|anygrasp_filtered) backend=$2 ;; *) fail "$usage" ;; esac
    shift 2
fi

if (( $# )); then
    action=$1
    shift
else
    printf '1) Get the machine feature ID from the AnyGrasp SDK\n2) Configure the AnyGrasp license and detection weights\nChoose an option [1/2]: '
    IFS= read -r action || fail 'No option entered. Choose 1 or 2.'
fi
case "$action" in
    1) (( $# == 0 )) || fail "$usage"
       task='Feature ID lookup' ;;
    2) (( $# <= 1 )) || fail "$usage"
       if (( $# == 0 )); then
           printf 'Enter the delivery directory, license ZIP, or checkpoint_detection.tar path: '
           IFS= read -r path || fail 'No delivery path entered.'
           set -- "$path"
       fi
       [[ -n "$1" ]] || fail 'The delivery path is empty.'
       task='License and detection weight configuration' ;;
    *) fail "Invalid option. Choose 1 or 2." ;;
esac

unset PYTHONPATH
export PYTHONNOUSERSITE=1
if [[ ${CONDA_DEFAULT_ENV:-} == rebot && -n ${CONDA_PREFIX:-} && -x ${CONDA_PREFIX:-}/bin/python ]]; then
    command=("$CONDA_PREFIX/bin/python")
elif conda_exe=$(find_conda); then
    command=("$conda_exe" run -n rebot --no-capture-output python)
else
    fail 'Conda was not found. Run bash scripts/setup_env.sh --backend anygrasp first.'
fi
# Keep native SDK diagnostics (including signature, expiry and machine mismatch)
# with the final status. Preserve the calling directory for relative input paths.
if output=$("${command[@]}" -B "$root/scripts/env/_anygrasp_license.py" "$backend" "$action" "$@" 2>&1); then
    printf '\n----\n%s\n[SUCCESS] %s succeeded.\n' "$output" "$task"
else
    status=$?
    if [[ $action == 2 && $status == 3 ]]; then
        printf '\n----\n%s\n[PARTIAL] License configured. Detection weights are missing.\n' "$output"
        exit 3
    fi
    fail "$task failed (exit code $status).
${output:-The SDK process returned no diagnostic information.}"
fi
