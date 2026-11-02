# Measurements

Numbers below were printed by processes in the amd64 container `lunar-mapping-dev` on this laptop. They are not estimates, and they are not Jetson timings. The Orin image was not built here.

## Library benchmark

`python3 /workspace/scripts/run_benchmark.py` on 1 November 2026. Eight synthetic scans, straight line at 0.3 m/s, 0.5 s apart, 90 by 8 rays, 1 cm range noise. Gates: at least 6 CLIPPER inliers, RMSE at most 0.12 m, fitness at least 0.5.

| Scan | Accepted | Reason | RMSE (m) | Inliers | Pose x (m) | Truth x (m) | Voxels | RSS (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | yes | initialized | 0 | 102 | 0.000 | 0.00 | 93 | 246.89 |
| 1 | yes | accepted | 0.0375 | 9 | 0.167 | 0.15 | 113 | 249.78 |
| 2 | yes | accepted | 0.0160 | 6 | 0.316 | 0.30 | 148 | 249.82 |
| 3 | no | below_min_inliers | — | 5 | 0.316 | 0.45 | 148 | 249.83 |
| 4 | yes | accepted | 0.0659 | 7 | 0.598 | 0.60 | 193 | 249.88 |
| 5 | yes | accepted | 0.0247 | 10 | 0.750 | 0.75 | 211 | 249.90 |
| 6 | no | below_min_inliers | — | 2 | 0.750 | 0.90 | 211 | 249.90 |
| 7 | no | below_min_inliers | — | 2 | 0.750 | 1.05 | 211 | 249.90 |

Summary from `logs/benchmark_summary.json`: 5 accepted, 3 rejected, 211 voxels, 33 static points, final RSS 249.90 MB. Mean input size 149.75 points. Mean preprocess 0.00766 s. Mean features 0.0555 s. Mean CLIPPER 0.0272 s over the seven scans that called it. Mean registration 0.231 s and mean map update 0.0526 s over the four scans that reached those steps. Final pose x 0.750 m against truth 1.05 m, because scans 6 and 7 were rejected and the pose stayed at the last accepted value. Rejected scans left the voxel count unchanged (148 after scan 3, 211 after scans 6 and 7).

## ROS launch

`ros2 launch dynamic_map pipeline.launch.py` in the same container, with `motion_delay_sec: 8` so the rover stays put until the nodes are up.

Over a 6.0 s window after tracking had already stopped, `/lidar/points` and `/mapping/processed_scan` each delivered 12 messages (2.000 Hz). `/mapping/registration_status` delivered 12 ERROR messages in that same window. `/mapping/global_map`, `/mapping/registered_scan`, and `/mapping/rover_pose` did not publish during that window, because those publishers run only for an accepted scan.

The four accepted updates, from `logs/metrics.jsonl`, were:

| Update | Points | Inliers | RMSE (m) | Voxels | Duration (s) | Features (s) | CLIPPER (s) | Registration (s) | Map (s) | RSS (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initialized | 113 | 102 | 0 | 108 | 0.198 | 0.0336 | — | — | — | 265.89 |
| stationary repeat | 113 | 95 | 0.000 | 118 | 0.316 | 0.0275 | 0.0790 | 0.0372 | 0.0591 | 269.26 |
| first motion | 114 | 13 | 0.0352 | 157 | 0.272 | 0.0581 | 0.0343 | 0.0549 | 0.0553 | 269.31 |
| second motion | 125 | 10 | 0.0345 | 199 | 0.372 | 0.0741 | 0.0257 | 0.138 | 0.0648 | 269.35 |

The stationary repeat is two scans of the same pose while `motion_delay_sec` was still holding the rover. After these four updates, later scans were rejected. A status sample after those rejections still reported 199 voxels, so the rejections did not change the map.

Mean synthetic scan generation after this launch started: 0.00337 s over 97 scans. Mean preprocessing: 0.0156 s over 87 scans.

Viser stayed up for the whole launch (no traceback in `logs/launch.txt`). From the Windows host, `http://localhost:8080` returned HTTP 200 and 1218 bytes. The viewer timer is the configured 8 Hz. This run did not measure a browser frame rate.

Cyclone DDS selected interface `lo`, reported that it is not multicast-capable, and disabled multicast. Discovery among the nodes in this one container still worked. A Windows ROS install was not used.

After rebuilding `lunar-mapping:humble`, `docker compose up -d dev` left `lunar-mapping-dev` running. Process 1 had sourced Humble (`AMENT_PREFIX_PATH` includes `/opt/ros/humble` and the workspace install). `scripts/check_pubsub.sh` printed `two-process pub/sub ok` and `pubsub ok phase1`.

## Tests

`python3 -m pytest` on the four package test directories: 36 passed in 7.98 s.

`python3 /opt/lunar/smoke_clipper.py` reported `clipperpy 0.2.4`, kept the six true inlier pairs, and printed `CLIPPER smoke test passed`.

The `lunar-mapping:humble` image build ran upstream CLIPPER `ctest`: 7 tests, 0 failed, 0.22 s. `CLIPPER.EuclideanDistance`, both get/set variants, the affinity test, `FindGlobalMax`, and both DSD tests passed. Eigen 3.3.9 was the CLIPPER dependency.

## Unitree probe

`bash /workspace/scripts/probe_unitree.sh` on 2 November 2026, inside the dev container, not inside the image build. It installed `libpcl-dev` and `ros-humble-pcl-conversions`, cloned `unitreerobotics/unilidar_sdk2`, and `colcon build` of `unitree_lidar_ros2` finished in 30.3 s. CMake printed a CMP0074 warning about `PCL_ROOT` and did not fail. The log is `logs/unitree_probe.txt` and ends with `BUILD_OK`.

That package is not installed in `lunar-mapping:humble`. The generic `unitree_adapter` node is still only a `PointCloud2` remap, and this laptop has no L2 to publish `/unilidar/cloud`.
