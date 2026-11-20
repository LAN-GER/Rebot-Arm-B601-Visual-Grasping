#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/scripts/_environment.sh"
action=check
env_options "$@"
cd "$root"
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
    if ! command -v "$tool" >/dev/null; then echo "[MISSING] Install system command: $tool"; host_missing=1; fi
done
if ! conda_exe=$(find_conda); then
    echo "[MISSING] Conda. Run ${next[*]} to install Miniforge and create rebot." >&2
    exit 1
fi
if ! version=$("$conda_exe" run -n rebot python -I -c '
import os, sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    sys.exit("No Python interpreter in rebot")
print(sys.version.split()[0])
'); then
    echo "[MISSING] The rebot environment has no working Python. Run: ${next[*]}" >&2
    exit 1
fi
if [[ "$version" != 3.10.* ]]; then
    echo "[MISSING] rebot has Python $version. The supported environment uses Python 3.10." >&2
    exit 1
fi
echo "[OK] rebot Python $version"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
status=0
"$conda_exe" run -n rebot --no-capture-output python -u scripts/_environment.py check "$backend" "$robot" || status=$?
case "$status" in
    0) ;;
    3) echo 'Supply the [MANUAL] items above, then rerun the checker.' >&2
       if (( host_missing )); then exit 1; fi
       exit 3 ;;
    *) echo "To install missing software: ${next[*]}" >&2; exit 1 ;;
esac
exit "$host_missing"
