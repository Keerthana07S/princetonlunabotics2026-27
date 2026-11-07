# Lunar rover LiDAR mapping

Python ROS 2 Humble pipeline for a hypothetical lunar rover. It registers consecutive LiDAR scans with the official CLIPPER data-association library and stores the result in a dynamic voxel map. The map, trajectories, clouds, and camera images are viewed in [Foxglove](https://foxglove.dev) through `foxglove_bridge`. The older Viser page is still available with `visualizer:=viser`.

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

Then view it in Foxglove, as described in the next section. The blue trail is ground truth and the red trail is the estimate. Map voxels go from blue (low confidence) to gray (high), and dynamic voxels are orange.

## Viewing in Foxglove

Every launch starts `foxglove_bridge` on `ws://localhost:8765` unless you pass `visualizer:=viser` (the Viser page on `http://localhost:8080`) or `visualizer:=none`.

1. Open the Foxglove desktop app or [app.foxglove.dev](https://app.foxglove.dev) in Chrome. [Lichtblick](https://github.com/lichtblick-suite/lichtblick), the open-source fork, also works.
2. Choose "Open connection", then "Foxglove WebSocket", with `ws://localhost:8765`.
3. Import the layout: Layouts menu, "Import from file", `config/foxglove/lunar_mapping.json`. It has a 3D view that follows `base_link`, the left camera image, and `/mapping/registration_status`.

The 3D view draws these topics:

- `/viz/map`: the voxel map with confidence colors, from `foxglove_helper`. Its `confidence` and `age` fields can be picked under "Color by" in the topic settings, with a color map, to see time since each voxel was last seen.
- `/mapping/registered_scan` (green): the last accepted scan in the map frame.
- `/lidar/points`: the raw sensor cloud colored by height, placed through TF.
- `/viz/ground_truth_path` (blue) and `/viz/estimated_path` (red).
- `/mapping/correspondences`: the CLIPPER inlier lines.

`foxglove_helper` (in `mapping_visualizer`) makes the `/viz/*` topics. It colors the map and keeps the two paths, with ground truth expressed relative to its first pose, which is where the map frame starts. In the LAC viewer, where nothing estimates a pose, it also broadcasts `map -> base_link` from ground truth and `base_link -> lidar_link` at the camera mount. Settings are in the `foxglove_bridge` and `foxglove_helper` sections of `config/visualization.yaml`. `python3 scripts/check_foxglove.py /viz/map ...` connects to the bridge as the app does, and reports which topics deliver messages.

Port 8765 has to be published by the container. A container created before this change only publishes 8080; recreate it with `docker compose up -d dev`.

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
- `config/visualization.yaml` Foxglove bridge and helper, and Viser rate and colors
- `config/foxglove/lunar_mapping.json` the Foxglove layout

Frames: [docs/frames.md](docs/frames.md). Topics: [docs/topics.md](docs/topics.md). DDS and Docker Desktop: [docs/networking.md](docs/networking.md).

## Packages

- `lidar_interface` converts `PointCloud2`, simulates a rover on terrain, relays an external cloud, turns a stereo pair into a cloud (`stereo_cloud`), and replays LAC logs (`lac_player`)
- `pointcloud_processing` filters scans, computes FPFH candidates, calls CLIPPER, and estimates a rigid transform
- `dynamic_map` keeps the voxel hash map and publishes pose, map, and registration status
- `mapping_visualizer` adds the Foxglove topics (`foxglove_helper`), holds the LAC viewer launch, and keeps the older Viser process

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
- `run_e2e.sh` runs the ROS launch, checks that the Foxglove bridge delivers the map, paths, scan, and TF (`logs/e2e_foxglove.json`; `VISUALIZER=viser` checks the Viser page instead), and calls `scripts/summarize_run.py`, which writes `logs/e2e_summary.json` from the per-node `logs/metrics_*.jsonl` files. There is one file per node, because concurrent appends to a single file on the Windows bind mount corrupted lines.

Results from this tree are in [docs/measurements.md](docs/measurements.md).

## Stereo pipeline

Stereo runs through the same pipeline as LiDAR. `stereo_cloud` turns a rectified image pair into a cloud on `/lidar/points`, and from there it is the LiDAR path: preprocessing, keypoints and FPFH, gated candidates, CLIPPER, Kabsch, ICP, the quality gate, the dynamic voxel map, and the visualizer.

```bash
docker compose exec dev bash -lc "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && ros2 launch dynamic_map stereo_pipeline.launch.py"
```

The launch replays `software/pointclouds/example (2).lac` with the `lac_player` node and starts the Foxglove bridge; the layout shows the left camera image next to the 3D view. Pass `visualizer:=viser` for the Viser page, `lac_path:=...` for another log, `playback_rate:=0.5` to slow it down, or `use_lac_player:=false` to feed live `/camera/left/image_raw` and `/camera/right/image_raw`.

Stereo values are in `config/stereo/`. Those files are loaded after the shared `config/*.yaml` and only list what differs. Compared with LiDAR, they change four things, and each was needed on the LAC log:

- `keypoint_method: random`. Voxel-grid keypoints from two scans of flat ground fall in the same grid cells, so CLIPPER finds a consistent "the rover did not move" set.
- `max_descriptor_distance: 1.0e9` with `gate_radius: 0.5`. Stereo FPFH distances between true matches were 5 to 80, so the LiDAR cap of 2.0 left no candidates. FPFH now only ranks pairs inside the motion gate.
- `coast_on_reject: true`. After a rejected pair, the pose advances by the constant-velocity prediction and the new pair becomes the reference. Without it, every scan after the first rejection was compared with a stale reference and also rejected. Rejected pairs are still never added to the map.
- The camera mount (`lidar_x/y/z`), a 0.1 m map voxel, and slower confidence decay, because a forward camera does not look back at what it has passed.

`scripts/run_stereo_benchmark.py` runs the same code and config without ROS and scores it against the log's rover pose. `scripts/run_stereo_e2e.sh` runs the launch, checks what the Foxglove bridge delivers (`logs/stereo_e2e_foxglove.json`), and writes `logs/stereo_e2e_summary.json`. Results are in [docs/measurements.md](docs/measurements.md#stereo-pipeline-on-the-lac-log).

`stereo_cloud` pairs images by stamp and publishes in the left camera's body frame (x forward, y left, z up). The defaults match the Lunar Autonomy Challenge front cameras: 1.22 rad horizontal field of view and a 0.162 m baseline. To build a map from a `.lac` log placed with the logged ground-truth poses, with no registration:

```bash
python3 /workspace/scripts/lac_stereo_map.py "/workspace/software/pointclouds/example (2).lac"
```

The script writes `logs/lac_stereo.ply`, a top-down height image `logs/lac_stereo.png`, and `logs/lac_stereo_summary.json`.

## Viewing a `.lac` log in ROS

To open a log in ROS and look at it in Foxglove, without any mapping:

```bash
docker compose exec dev bash /workspace/scripts/play_lac.sh "/workspace/software/pointclouds/example (2).lac" rate:=2.0 loop:=true
```

This runs `ros2 launch mapping_visualizer lac_view.launch.py`. `lac_player` publishes the left and right images, `/ground_truth/pose`, and `/imu/data` in mission time. `stereo_cloud` turns each pair into a cloud on `/lidar/points`. In Foxglove, with the layout above, you see the left image, the ground-truth path, and the current stereo cloud on the rover at the camera mount, colored by height. `scripts/play_lac_viser.sh` shows the same thing in Viser. Loading takes about 10 s, because the first read indexes the whole archive. `scripts/lac_publish_only.sh` publishes the topics with no viewer, for `rviz2`, `ros2 topic echo`, or your own nodes.

To get a normal rosbag2 that `ros2 bag play`, `ros2 bag info`, and other tools can open, convert the log:

```bash
ros2 run lidar_interface lac_to_bag "/workspace/software/pointclouds/example (2).lac" /workspace/logs/example_bag --stereo
ros2 launch mapping_visualizer lac_view.launch.py log:=/workspace/logs/example_bag stereo:=false
```

`--stereo` also writes the stereo clouds to `/lidar/points`, so playback needs no stereo matching; pass `stereo:=false` to the viewer in that case so the clouds are not published twice. The example log converts in about 40 s to a 117 MiB bag holding 60 image pairs, 60 clouds, and 1182 pose and IMU samples over 59 s. If you leave out the output path, the bag goes next to the log. `--all-rows` keeps the rows before the first image, and `software/pointclouds/lac_to_ros2bag.py` is the same converter.

## Boulder traverses

`traverse_*.csv` rows are `frame,x,y,z,detections,boulders`: the rover position and the detected boulder centroids, both in the world frame, with no heading. `pointcloud_processing.boulder_association` treats the boulders as landmarks. It drives a drifting odometry estimate along the logged path, pairs each frame's detections with nearby map landmarks, keeps CLIPPER's consistent set, and fits a yaw-and-translation correction. Ground-truth boulder identities come from clustering all detections within 0.15 m, so every candidate pair can be scored as correct or wrong.

```bash
python3 /workspace/scripts/run_boulder_benchmark.py "/workspace/traverse_1 (1).csv"
```

writes `logs/boulder_benchmark_<name>.jsonl` and `_summary.json`. To watch a traverse in Foxglove:

```bash
ros2 launch mapping_visualizer boulder_clipper.launch.py csv:="/workspace/traverse_1 (1).csv" rate:=5.0
```

Open `config/foxglove/boulder_clipper.json`. Playback starts when Foxglove subscribes. `/boulders/associations` draws each candidate pair from the detection (at the predicted pose) to the landmark, in four namespaces: `kept_correct` (green), `kept_wrong` (red), `dropped_correct` (orange), and `dropped_wrong` (grey, hidden by default). `/boulders/status` has the per-frame counts and the accept or reject reason, and the plot compares the corrected and odometry-only errors. `start_frame:=3000` skips ahead, `seed:=` changes the odometry noise, and `correct:=false` runs odometry only.

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
- Stereo odometry is geometry only. The front camera mostly sees smooth ground, which constrains height and tilt but barely constrains forward motion and heading. On the LAC log, accepted pairs have a median rotation error of about 0.75 degrees, so heading and pitch drift. The final error ranged from 2.4 m offline to 7.7 m in the ROS launch over 17.5 m. Small differences such as float32 messages change which pairs are accepted.
- Open3D and CLIPPER are not bit-for-bit repeatable here. Two identical runs of the synthetic LiDAR benchmark accepted 64 and 57 of 69 scans.
- No IMU fusion. The odometry message can accept one later. The mapper does not read `/sim/ground_truth_pose`.
- The synthetic scanner is a ray caster, not an L2 beam pattern.
- Open3D and CLIPPER run on CPU.
- Docker Desktop DDS is tested inside one container, not from a ROS install on Windows.
