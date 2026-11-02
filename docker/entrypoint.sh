#!/usr/bin/env bash
# ROS setup scripts read optional variables, so nounset stays off.
set -eo pipefail

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if [[ -f /workspace/ros2_ws/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /workspace/ros2_ws/install/setup.bash
fi

exec "$@"
