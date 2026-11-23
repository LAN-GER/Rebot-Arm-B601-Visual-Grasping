#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/scripts/env/_environment.sh"
action=check
env_options "$@"
cd "$root"
summary_dir=$(mktemp -d)
record() { printf '%s\n' "$2" >> "$summary_dir/$1"; }

print_group() {
    local file="$summary_dir/$1" title=$2
    local -a items=()
    if [[ -s "$file" ]]; then mapfile -t items < "$file"; fi
    printf '%s (%d)\n' "$title" "${#items[@]}"
    if (( ${#items[@]} )); then printf '  - %s\n' "${items[@]}"; fi
}
print_summary() {
    printf '\n----\nEnvironment check summary / 环境检查结果\n'
    print_group ready 'Installed or ready / 已安装或已就绪'
    print_group missing 'Missing or unusable / 缺失或不可用'
    if [[ -s "$summary_dir/manual" ]]; then print_group manual 'Files to supply / 待手动提供'; fi
    if [[ -s "$summary_dir/skipped" ]]; then print_group skipped 'Not checked / 未检查'; fi
    rm -r -- "$summary_dir"
}
trap print_summary EXIT
printf 'Project environment check | rebot | backend=%s | B601=%s\n' "$backend" "$robot"
next=(bash scripts/setup_env.sh --backend "$backend")
if (( robot )) && [[ "$backend" != all ]]; then next+=(--robot); fi

if [[ "$backend" != obb ]]; then
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L; then
        echo '[NOTICE] Check the NVIDIA driver for the GPU backends. OBB can use a CPU.'
    fi
fi
host_missing=0
for tool in git curl; do
    if command -v "$tool" >/dev/null; then
        record ready "$tool"
    else
        echo "[MISSING] Install system command: $tool"
        record missing "$tool"
        host_missing=1
    fi
done
if ! conda_exe=$(find_conda); then
    record missing 'Conda'
    record skipped 'Packages and SDKs in rebot (Conda unavailable)'
    echo "[MISSING] Conda. Run ${next[*]} to install Miniforge and create rebot." >&2
    exit 1
fi
record ready 'Conda executable'
if ! version=$("$conda_exe" run -n rebot python -I -c '
import os, sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    sys.exit("No Python interpreter in rebot")
print(sys.version.split()[0])
'); then
    record missing 'rebot environment with Python 3.10'
    record skipped 'Packages and SDKs in rebot (Python unavailable)'
    echo "[MISSING] The rebot environment has no working Python. Run: ${next[*]}" >&2
    exit 1
fi
if [[ "$version" != 3.10.* ]]; then
    record missing "Python 3.10 in rebot (found $version)"
    record skipped 'Packages and SDKs in rebot (unsupported Python version)'
    echo "[MISSING] rebot has Python $version. The supported environment uses Python 3.10." >&2
    exit 1
fi
record ready "rebot Python $version"
echo "[OK] rebot Python $version"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
status=0
"$conda_exe" run -n rebot --no-capture-output python -u scripts/env/_environment.py check "$backend" "$robot" "$summary_dir" || status=$?
if (( status != 0 )) && [[ ! -f "$summary_dir/finished" ]]; then
    record skipped 'The package and SDK checks did not finish'
fi
case "$status" in
    0) ;;
    3) echo 'Supply the [MANUAL] items above, then rerun the checker.' >&2
       if (( host_missing )); then exit 1; fi
       exit 3 ;;
    *) echo "To install missing software: ${next[*]}" >&2; exit 1 ;;
esac
exit "$host_missing"
