#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../examples/camera_check/run.sh" "$@"
