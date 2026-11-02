# Lunar rover LiDAR mapping

Python ROS 2 Humble pipeline for a hypothetical lunar rover. It registers consecutive LiDAR scans with the official CLIPPER data-association library and stores the result in a dynamic voxel map. A separate process draws the map in Viser.

This is LiDAR odometry and a local map. It does not close loops, and it does not remove drift. There is no GPS.

The official computer is an NVIDIA Jetson Orin Nano running JetPack 6 (Ubuntu 22.04, aarch64). The same CPU image is what you build on a Windows laptop. No NVIDIA GPU is required.

The Unitree L2 is not assumed. Anything vendor-specific is the `unitree_adapter` remap, which only forwards `sensor_msgs/PointCloud2`. The SDK is not installed in the image. Upstream documents that driver for Ubuntu 20.04 and ROS 2 Foxy.

## Laptop

Install Docker Desktop and start it. From this repository:

```bash
docker compose build dev
docker compose up -d dev
docker compose exec dev bash -lc "cd /workspace/ros2_ws && colcon build"
docker compose exec dev bash -lc "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && cd /workspace/ros2_ws && colcon test --event-handlers console_direct+"
```

Launch the synthetic pipeline inside the dev container:

```bash
docker compose exec dev bash -lc "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && ros2 launch dynamic_map pipeline.launch.py"
```

Then open `http://localhost:8080`. The blue trail is ground truth and the red trail is the estimate. Use the "Color by" dropdown to switch the map between confidence and time since each voxel was last seen.

`OMP_NUM_THREADS=4` is set in the image and in `docker-compose.yml`. Without it, Open3D starts one thread per core, and on this laptop each scan took about 5 times longer. A container created before that change needs `docker exec -e OMP_NUM_THREADS=4`.

Cursor can attach with the dev container in `.devcontainer/`. The workspace inside the container is `/workspace`.

`config/`, `bags/`, and `logs/` are on the bind mount. The OneDrive path makes that mount slower than a normal disk.

## Jetson Orin Nano

Use JetPack 6. Build on the Orin. Do not load the laptop's amd64 image.

```bash
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build dev
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d dev
```

An 8 GB module is the practical target. A 4 GB module can run the synthetic demo if you raise `voxel_size` and lower `azimuth_bins` in `config/`.

## What is configurable

Parameters that need a real sensor are marked `TUNE FOR REAL SENSOR` in `config/`.

- `config/lidar.yaml` range, field of view, scan rate, noise
- `config/preprocessing.yaml` voxel size and outlier removal
- `config/features.yaml` keypoints and FPFH
- `config/clipper.yaml` geometric consistency (`sigma`, `epsilon`)
- `config/registration.yaml` quality gates, ICP, `scan_to_scan` or `scan_to_submap`
- `config/mapping.yaml` voxel resolution, confidence decay, free-space clearing
- `config/visualization.yaml` Viser rate and colors

Frames: [docs/frames.md](docs/frames.md). Topics: [docs/topics.md](docs/topics.md). DDS and Docker Desktop: [docs/networking.md](docs/networking.md).

## Packages

- `lidar_interface` converts `PointCloud2`, simulates a rover on terrain, relays an external cloud, and turns a stereo pair into a cloud (`stereo_cloud`)
- `pointcloud_processing` filters scans, computes FPFH candidates, calls CLIPPER, and estimates a rigid transform
- `dynamic_map` keeps the voxel hash map and publishes pose, map, and registration status
- `mapping_visualizer` is the Viser process

Algorithms do not import `rclpy`. The nodes do.

A rejected registration publishes `diagnostic_msgs/DiagnosticStatus` level ERROR and does not change the stored voxels.

## CLIPPER

The image builds [mit-acl/clipper](https://github.com/mit-acl/clipper) at commit `e514dc29c273837ffdfeebbefbdcb2a93d970969` and installs the `clipperpy` module. Eigen 3.3.9 is used because Eigen 3.4 on Ubuntu 22.04 has an upstream double-free report. The upstream `-mavx -mfma` flags are removed so the same sources compile on the Orin. The optional PMC maximum-clique solver is off. The SCS semidefinite solver is built with system BLAS and LAPACK so the upstream tests that use it can run. The mapping pipeline calls `CLIPPER.solve()` with the Euclidean-distance invariant, which is the projected-gradient association from the paper.

## Tests and measurements

```bash
docker compose exec dev bash -lc "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && python3 -m pytest -q /workspace/ros2_ws/src"
docker compose exec dev bash -lc "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && python3 /workspace/scripts/run_benchmark.py"
docker compose exec dev bash -lc "bash /workspace/scripts/run_e2e.sh"
```

- `run_benchmark.py` drives the whole synthetic path without ROS, using the parameters in `config/`. Override a parameter with `--set key=value`. It scores the result against ground truth, counts voxels on the objects that appear, disappear, and move, and writes `logs/benchmark_trajectory.jsonl` and `logs/benchmark_trajectory_summary.json`.
- `run_e2e.sh` runs the ROS launch, checks the Viser page, and calls `scripts/summarize_run.py`, which writes `logs/e2e_summary.json` from the per-node `logs/metrics_*.jsonl` files. There is one file per node, because concurrent appends to a single file on the Windows bind mount corrupted lines.

Results from this tree are in [docs/measurements.md](docs/measurements.md).

## LAC stereo

`ros2 run lidar_interface stereo_cloud` subscribes to a left and a right `sensor_msgs/Image` (`left_topic` and `right_topic`, default `/camera/left/image_raw` and `/camera/right/image_raw`), pairs them by stamp, and publishes `/lidar/points` in the left camera's body frame (x forward, y left, z up). The defaults match the Lunar Autonomy Challenge front cameras: 1.22 rad horizontal field of view and a 0.162 m baseline. To build a ground-truth-posed map from a `.lac` log without ROS:

```bash
python3 /workspace/scripts/lac_stereo_map.py "/workspace/software/pointclouds/example (2).lac"
```

The script writes `logs/lac_stereo.ply`, a top-down height image `logs/lac_stereo.png`, and `logs/lac_stereo_summary.json`.

## Unitree L2

`ros2 run lidar_interface unitree_adapter` subscribes to `/unilidar/cloud` and republishes `/lidar/points`. It does not link the vendor SDK. To try the upstream ROS 2 package on Humble:

```bash
docker compose exec dev bash /workspace/scripts/probe_unitree.sh
```

The log is `logs/unitree_probe.txt`. A failed build leaves the generic interface in place.

## Limitations

- No loop closure and no global pose-graph correction. On the 9 m synthetic drive, the final error was 0.34 m offline and 0.40 m in the ROS launch, mostly as an underestimate of forward motion.
- A ray that returns nothing (sky) gives no free-space evidence, so a removed object only clears when there is a surface behind it.
- Scans published before the mapping node subscribes are dropped. The synthetic end-to-end latency includes the time spent ray casting the scan.
- The stereo node uses the documented LAC camera geometry, not a calibration file, and has not been run against a live LAC simulator.
- No IMU fusion. The odometry message can accept one later. The mapper does not read `/sim/ground_truth_pose`.
- The synthetic scanner is a ray caster, not an L2 beam pattern.
- Open3D and CLIPPER run on CPU.
- Docker Desktop DDS is tested inside one container, not from a ROS install on Windows.
