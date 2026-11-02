#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
exec python3 /workspace/software/pointclouds/lac_to_ros2bag.py "/workspace/software/pointclouds/example (2).lac"
