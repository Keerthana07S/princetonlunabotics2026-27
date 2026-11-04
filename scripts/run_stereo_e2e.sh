#!/usr/bin/env bash
# Run the stereo pipeline launch on a LAC log until playback finishes (or
# TIMEOUT_S passes), then summarize the metrics it wrote.
# Usage: [VISUALIZER=foxglove|viser] scripts/run_stereo_e2e.sh [LAC_PATH] [TIMEOUT_S]
set -eo pipefail
set -m
lac_path="${1:-/workspace/software/pointclouds/example (2).lac}"
timeout_s="${2:-180}"
visualizer="${VISUALIZER:-foxglove}"
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p /workspace/logs
log=/workspace/logs/stereo_e2e_launch.txt
since="$(date +%s.%N)"
ros2 launch dynamic_map stereo_pipeline.launch.py "lac_path:=${lac_path}" "visualizer:=${visualizer}" > "${log}" 2>&1 &
launch_pid=$!
sleep 15
if [ "${visualizer}" = "viser" ]; then
  viser_status="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ || true)"
  echo "Viser HTTP status after 15 s: ${viser_status}"
else
  # The first map arrives after the LAC index load and two stereo pairs.
  sleep 15
  python3 /workspace/scripts/check_foxglove.py --seconds 10 \
    /viz/map /viz/estimated_path /viz/ground_truth_path /camera/left/image_raw /lidar/points /tf \
    > /workspace/logs/stereo_e2e_foxglove.json || true
  echo "Foxglove bridge check after 30 s: logs/stereo_e2e_foxglove.json"
fi
for _ in $(seq 15 "${timeout_s}"); do
  if grep -q "Playback finished" "${log}"; then
    break
  fi
  sleep 1
done
# Let the mapper finish the last pair.
sleep 5
kill -INT "${launch_pid}"
wait "${launch_pid}" || true
python3 /workspace/scripts/summarize_run.py --since "${since}" --output /workspace/logs/stereo_e2e_summary.json
