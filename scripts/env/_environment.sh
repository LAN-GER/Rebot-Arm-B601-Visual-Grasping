# Common options for the two project environment commands.
env_usage() {
    cat <<HELP
Usage: bash scripts/${action}_env.sh [--backend all|obb|anygrasp|graspnet] [--robot]
  Default: all backends, the configured RGB-D camera, YOLOE and the B601 SDK.
  A single backend selects camera preview dependencies; add --robot for the B601.
HELP
    if [[ "$action" == setup ]]; then
        echo '  --checkpoint-file PATH   Supply a local GraspNet-Baseline checkpoint.'
        echo '  Creates or repairs rebot and downloads missing public SDKs. If Conda is absent,'
        echo '  installs Miniforge in ~/miniforge3.'
    else
        echo '  Inspects rebot without installing packages or connecting to hardware.'
    fi
}
env_options() {
    backend=all
    robot=0
    checkpoint_file=""
    while (( $# )); do
        case "$1" in
            -h|--help) env_usage; exit 0 ;;
            --backend)
                case "${2:-}" in all|obb|anygrasp|graspnet) backend=$2 ;; *) env_usage >&2; exit 2 ;; esac
                shift 2 ;;
            --robot) robot=1; shift ;;
            --checkpoint-file)
                if [[ "$action" != setup || ! -s "${2:-}" || ! -f "${2:-}" ]]; then
                    env_usage >&2; exit 2
                fi
                checkpoint_file=$(realpath -- "$2")
                shift 2 ;;
            *) env_usage >&2; exit 2 ;;
        esac
    done
    if [[ "$backend" == all ]]; then robot=1; fi
    if [[ -n "$checkpoint_file" && "$backend" != all && "$backend" != graspnet ]]; then
        echo '--checkpoint-file requires --backend graspnet or all' >&2; exit 2
    fi
}

find_conda() {
    local candidate
    for candidate in "${CONDA_EXE:-conda}" "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda" "$HOME/anaconda3/bin/conda"; do
        if command -v "$candidate" >/dev/null; then command -v "$candidate"; return; fi
    done
    return 1
}
