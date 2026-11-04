#!/usr/bin/env bash
# View a .lac recording (or a bag from lac_to_bag) in Foxglove: connect to
# ws://localhost:8765 and open config/foxglove/lunar_mapping.json.
# Usage: scripts/play_lac.sh [LOG] [extra launch args, e.g. rate:=2.0 loop:=true visualizer:=viser]
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
log="${1:-/workspace/software/pointclouds/example (2).lac}"
shift || true
exec ros2 launch mapping_visualizer lac_view.launch.py "log:=${log}" "$@"
