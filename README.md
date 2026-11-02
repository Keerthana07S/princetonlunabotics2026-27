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

Then open `http://localhost:8080`.

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

- `lidar_interface` converts `PointCloud2`, simulates a rover, and relays an external cloud
- `pointcloud_processing` filters scans, computes FPFH candidates, calls CLIPPER, and estimates a rigid transform
- `dynamic_map` keeps the voxel hash map and publishes pose, map, and registration status
- `mapping_visualizer` is the Viser process

Algorithms do not import `rclpy`. The nodes do.

A rejected registration publishes `diagnostic_msgs/DiagnosticStatus` level ERROR and does not change the stored voxels.

## CLIPPER

The image builds [mit-acl/clipper](https://github.com/mit-acl/clipper) at commit `e514dc29c273837ffdfeebbefbdcb2a93d970969` and installs the `clipperpy` module. Eigen 3.3.9 is used because Eigen 3.4 on Ubuntu 22.04 has an upstream double-free report. The upstream `-mavx -mfma` flags are removed so the same sources compile on the Orin. The optional PMC maximum-clique solver is off. The SCS semidefinite solver is built with system BLAS and LAPACK so the upstream tests that use it can run. The mapping pipeline calls `CLIPPER.solve()` with the Euclidean-distance invariant, which is the projected-gradient association from the paper.

## Tests and measurements

```bash
docker compose exec dev bash -lc "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && python3 -m pytest /workspace/ros2_ws/src/lidar_interface/test /workspace/ros2_ws/src/pointcloud_processing/test /workspace/ros2_ws/src/dynamic_map/test /workspace/ros2_ws/src/mapping_visualizer/test"
docker compose exec dev bash -lc "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && python3 /workspace/scripts/run_benchmark.py"
```

Benchmark rows are appended to `logs/benchmark.jsonl` by that script. Summarize them from the file rather than copying numbers by hand into a new experiment. The measurements from the run that produced this tree are in [docs/measurements.md](docs/measurements.md) after that script has been executed.

## Unitree L2

`ros2 run lidar_interface unitree_adapter` subscribes to `/unilidar/cloud` and republishes `/lidar/points`. It does not link the vendor SDK. To try the upstream ROS 2 package on Humble:

```bash
docker compose exec dev bash /workspace/scripts/probe_unitree.sh
```

The log is `logs/unitree_probe.txt`. A failed build leaves the generic interface in place.

## Limitations

- No loop closure and no global pose-graph correction.
- No IMU fusion. The odometry message can accept one later. The mapper does not read `/sim/ground_truth_pose`.
- The synthetic scanner is a ray caster, not an L2 beam pattern.
- Open3D and CLIPPER run on CPU.
- Docker Desktop DDS is tested inside one container, not from a ROS install on Windows.
