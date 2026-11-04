#!/usr/bin/env python3
"""Convert a .lac log into a ROS 2 bag. Same as ``ros2 run lidar_interface lac_to_bag``.

    python3 lac_to_ros2bag.py "example (2).lac" example_bag [--stereo]

To publish a .lac live instead, use ``ros2 run lidar_interface lac_player
--ros-args -p lac_path:=<file>``, or view it with
``ros2 launch lidar_interface lac_view.launch.py log:=<file>``.
Source /opt/ros/humble and /workspace/ros2_ws/install first.
"""

from lidar_interface.lac_bag import main

if __name__ == "__main__":
    main()
