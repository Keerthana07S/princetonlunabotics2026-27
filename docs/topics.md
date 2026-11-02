# ROS topics

Sensor streams use the default sensor QoS profile: best effort, keep last, depth 5. `ros2 topic echo` on those topics needs `--qos-reliability best_effort`.

| Topic | Type | QoS | Publisher |
| --- | --- | --- | --- |
| `/lidar/points` | `sensor_msgs/PointCloud2` | sensor | synthetic LiDAR, cloud relay, or Unitree adapter |
| `/mapping/processed_scan` | `sensor_msgs/PointCloud2` | sensor | preprocessing |
| `/mapping/registered_scan` | `sensor_msgs/PointCloud2` | sensor | mapping, map frame, only after an accepted registration |
| `/mapping/global_map` | `sensor_msgs/PointCloud2` | reliable, depth 1 (configurable) | mapping. `intensity` is confidence, `label` is 1 for dynamic voxels |
| `/mapping/rover_pose` | `nav_msgs/Odometry` | reliable, depth 10 | mapping. `frame_id` is `map`, `child_frame_id` is `base_link` |
| `/mapping/registration_status` | `diagnostic_msgs/DiagnosticStatus` | reliable, depth 10 | mapping. ERROR leaves the previous map in place |
| `/mapping/correspondences` | `visualization_msgs/Marker` LINE_LIST | reliable, depth 10 | mapping, inlier pairs in the map frame |
| `/sim/ground_truth_pose` | `geometry_msgs/PoseStamped` | reliable, depth 10 | synthetic LiDAR only |

TF: `map` to `base_link`, and `base_link` to `lidar_link`.

Names and frames are set in `config/`.
