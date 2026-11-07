# ROS topics

Sensor streams use the default sensor QoS profile: best effort, keep last, depth 5. `ros2 topic echo` on those topics needs `--qos-reliability best_effort`.

| Topic | Type | QoS | Publisher |
| --- | --- | --- | --- |
| `/lidar/points` | `sensor_msgs/PointCloud2` | sensor | synthetic LiDAR, cloud relay, Unitree adapter, or stereo cloud |
| `/camera/left/image_raw`, `/camera/right/image_raw` | `sensor_msgs/Image` mono8 | reliable, depth 10 | LAC player (or a live camera driver). Both images of a pair carry the same stamp |
| `/ground_truth/pose` | `geometry_msgs/PoseStamped` | reliable, depth 10 | LAC player or a `lac_to_bag` bag, rover pose in the LAC world frame |
| `/imu/data` | `sensor_msgs/Imu` | reliable, depth 10 | LAC player or a `lac_to_bag` bag. Nothing subscribes yet |
| `/mapping/processed_scan` | `sensor_msgs/PointCloud2` | sensor | preprocessing |
| `/mapping/registered_scan` | `sensor_msgs/PointCloud2` | sensor | mapping, map frame, only after an accepted registration |
| `/mapping/global_map` | `sensor_msgs/PointCloud2` | reliable, depth 1 (configurable) | mapping. `intensity` is confidence, `label` is 1 for dynamic voxels |
| `/mapping/rover_pose` | `nav_msgs/Odometry` | reliable, depth 10 | mapping. `frame_id` is `map`, `child_frame_id` is `base_link`. With `coast_on_reject`, rejected scans also publish the coasted pose |
| `/mapping/registration_status` | `diagnostic_msgs/DiagnosticStatus` | reliable, depth 10 | mapping. ERROR leaves the previous map in place |
| `/mapping/correspondences` | `visualization_msgs/Marker` LINE_LIST | reliable, depth 10 | mapping, inlier pairs in the map frame |
| `/sim/ground_truth_pose` | `geometry_msgs/PoseStamped` | reliable, depth 10 | synthetic LiDAR only |
| `/viz/map` | `sensor_msgs/PointCloud2` | same as `/mapping/global_map` | foxglove_helper. Fields `x y z rgb confidence dynamic age`; `rgb` is packed `0x00RRGGBB` uint32 with confidence colors and dynamic voxels in orange |
| `/viz/ground_truth_path`, `/viz/estimated_path` | `nav_msgs/Path` | reliable, depth 1 | foxglove_helper, map frame, a pose every 0.05 m. Ground truth is relative to its first pose |
| `/viz/ground_truth_pose` | `geometry_msgs/PoseStamped` | reliable, depth 10 | foxglove_helper, ground truth in the map frame |

The boulder traverse viewer (`boulder_clipper.launch.py`) publishes its own topics, all in the `map` frame, which is the CSV world frame:

| Topic | Type | QoS | Publisher |
| --- | --- | --- | --- |
| `/boulders/associations` | `visualization_msgs/MarkerArray` LINE_LIST | reliable, depth 10 | boulder_clipper. Namespaces `kept_correct`, `kept_wrong`, `dropped_correct`, `dropped_wrong` |
| `/boulders/detections` | `visualization_msgs/MarkerArray` SPHERE_LIST | reliable, depth 10 | boulder_clipper. `predicted` (before correction), `corrected`, `logged` (CSV world positions) |
| `/boulders/landmarks` | `visualization_msgs/MarkerArray` | reliable, depth 10 | boulder_clipper. `landmarks` before this frame's update, `ground_truth_boulders`, and `labels` with ground-truth identities |
| `/boulders/truth_path`, `/boulders/odometry_path`, `/boulders/estimated_path` | `nav_msgs/Path` | reliable, depth 1 | boulder_clipper, one pose per detection frame |
| `/boulders/status` | `diagnostic_msgs/DiagnosticStatus` | reliable, depth 10 | boulder_clipper. WARN when the frame's correction is rejected |
| `/boulders/translation_error`, `/boulders/odometry_translation_error` | `std_msgs/Float64` | reliable, depth 10 | boulder_clipper, meters |

It broadcasts `map` to `base_link` at the estimated pose and `map` to `truth_link` at the logged position.

TF: `map` to `base_link`, and `base_link` to `lidar_link`. The mapping node publishes both. In the LAC viewer, which has no mapping node, `foxglove_helper` publishes `map` to `base_link` from ground truth and a static `base_link` to `lidar_link` at the camera mount.

Names and frames are set in `config/`.
