#!/usr/bin/env bash
# View a .lac recording (or a bag from lac_to_bag) in the older Viser page at http://localhost:8080.
# Usage: scripts/play_lac_viser.sh [LOG] [extra launch args, e.g. rate:=2.0 loop:=true]
set -eo pipefail
log="${1:-/workspace/software/pointclouds/example (2).lac}"
shift || true
exec /workspace/scripts/play_lac.sh "${log}" visualizer:=viser "$@"
