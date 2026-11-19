#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
exec bash "$root/scripts/run_python.sh" -m rebot_grasp record "$@"
