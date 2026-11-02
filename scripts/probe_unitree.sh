#!/usr/bin/env bash
# Optional probe. The default image does not install the Unitree SDK.
# Official unilidar_sdk2 docs list Ubuntu 20.04 and ROS 2 Foxy, not Humble.
set -eo pipefail
LOG=/workspace/logs/unitree_probe.txt
mkdir -p /workspace/logs
{
  echo "probe_start $(date -Is)"
  echo "ros_distro ${ROS_DISTRO:-unset}"
  source /opt/ros/humble/setup.bash
  apt-get update
  apt-get install -y --no-install-recommends libpcl-dev ros-humble-pcl-conversions || echo "APT_EXIT:$?"
  rm -rf /tmp/unilidar_sdk2
  if ! git clone --depth 1 https://github.com/unitreerobotics/unilidar_sdk2.git /tmp/unilidar_sdk2; then
    echo "CLONE_FAILED"
    exit 0
  fi
  mapfile -t candidates < <(find /tmp/unilidar_sdk2 -type d -name unitree_lidar_ros2)
  echo "candidates ${candidates[*]:-none}"
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "NO_ROS2_PACKAGE"
    exit 0
  fi
  cd "${candidates[0]}"
  if colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release; then
    echo "BUILD_OK"
  else
    echo "BUILD_FAILED"
  fi
} 2>&1 | tee "$LOG"
echo "wrote $LOG"
