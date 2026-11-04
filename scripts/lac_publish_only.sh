#!/usr/bin/env bash
# Publish a .lac recording on the camera, ground-truth, and IMU topics, with no viewer.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
log="${1:-/workspace/software/pointclouds/example (2).lac}"
exec ros2 run lidar_interface lac_player --ros-args -p "lac_path:=${log}" -p "wait_for_topics:=['']"
