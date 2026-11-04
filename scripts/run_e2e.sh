#!/usr/bin/env bash
# Run the synthetic pipeline launch for DURATION seconds, then summarize the
# metrics it wrote. Usage: [VISUALIZER=foxglove|viser] scripts/run_e2e.sh [DURATION_S]
set -eo pipefail
# Job control gives the background launch (and its nodes) normal SIGINT handling.
set -m
duration="${1:-50}"
visualizer="${VISUALIZER:-foxglove}"
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
mkdir -p /workspace/logs
since="$(date +%s.%N)"
ros2 launch dynamic_map pipeline.launch.py "visualizer:=${visualizer}" > /workspace/logs/e2e_launch.txt 2>&1 &
launch_pid=$!
sleep 15
if [ "${visualizer}" = "viser" ]; then
  viser_status="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ || true)"
  echo "Viser HTTP status after 15 s: ${viser_status}"
  sleep "$((duration - 15))"
else
  python3 /workspace/scripts/check_foxglove.py --seconds 10 \
    /viz/map /viz/estimated_path /viz/ground_truth_path /mapping/registered_scan /tf \
    > /workspace/logs/e2e_foxglove.json || true
  echo "Foxglove bridge check after 15 s: logs/e2e_foxglove.json"
  sleep "$((duration - 25))"
fi
kill -INT "${launch_pid}"
wait "${launch_pid}" || true
python3 /workspace/scripts/summarize_run.py --since "${since}"
