#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../examples/obb_filtered/run.sh" "$@"
