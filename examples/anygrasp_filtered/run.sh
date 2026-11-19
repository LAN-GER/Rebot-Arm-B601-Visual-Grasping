#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
for arg in "$@"; do
    case "$arg" in
        -h|--help|--output|--output=*) exec bash "$root/scripts/run_python.sh" -m rebot_grasp anygrasp-filtered "$@" ;;
    esac
done
mkdir -p "$root/outputs/anygrasp-filtered"
session=$(mktemp -d "$root/outputs/anygrasp-filtered/run-XXXXXXXX")
echo "Run directory: $session"
bash "$root/scripts/run_python.sh" -m rebot_grasp anygrasp-filtered --output "$session/result" "$@" 2>&1 | tee "$session/console.log"
