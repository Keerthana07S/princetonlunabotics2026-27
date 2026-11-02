#!/usr/bin/env bash
# Play a .lac recording into the Viser process. example.lac in this repo is empty;
# pass the real archive as the first argument.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash

lac_path="${1:-/workspace/software/pointclouds/example (2).lac}"
mkdir -p /workspace/logs

ros2 run mapping_visualizer viser_visualizer > /workspace/logs/lac_viser.txt 2>&1 &
viser_pid=$!
sleep 3
python3 /workspace/software/pointclouds/lac_to_ros2bag.py "$lac_path" > /workspace/logs/lac_pub.txt 2>&1 &
pub_pid=$!
echo "viser_pid=$viser_pid publisher_pid=$pub_pid"
echo "publisher started for $lac_path"
wait "$viser_pid" "$pub_pid"
