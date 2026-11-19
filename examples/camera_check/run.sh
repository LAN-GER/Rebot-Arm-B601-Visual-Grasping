#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
for arg in "$@"; do
    case "$arg" in
        -h|--help|--output|--output=*) exec bash "$root/scripts/run_python.sh" -m rebot_grasp camera-check --view "$@" ;;
    esac
done
mkdir -p "$root/outputs/camera-check"
session=$(mktemp -d "$root/outputs/camera-check/run-XXXXXXXX")
echo "Run directory: $session"
bash "$root/scripts/run_python.sh" -m rebot_grasp camera-check --view --output "$session/result" "$@" 2>&1 | tee "$session/console.log"
