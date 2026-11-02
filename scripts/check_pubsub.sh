#!/usr/bin/env bash
# Two processes in this container. ROS_LOCALHOST_ONLY=1 is the tested Docker Desktop path.
set -eo pipefail
source /opt/ros/humble/setup.bash
rm -f /tmp/lunar_echo.txt
ros2 topic echo /lunar/smoke std_msgs/msg/String --once > /tmp/lunar_echo.txt &
echo_pid=$!
sleep 2
ros2 topic pub /lunar/smoke std_msgs/msg/String "{data: phase1}" --once
wait "$echo_pid"
if ! grep -q "phase1" /tmp/lunar_echo.txt; then
  echo "two-process echo did not see phase1" >&2
  cat /tmp/lunar_echo.txt >&2 || true
  exit 1
fi
echo "two-process pub/sub ok"
python3 /workspace/scripts/check_pubsub.py
