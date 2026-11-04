# Measurements

All numbers below were produced on this laptop by processes running in the amd64 container `lunar-mapping-dev`. None are estimates, and none are Jetson timings: the Orin image was not built here. Every run set `OMP_NUM_THREADS=4`. With Open3D's default thread count (22 threads on this laptop), the same offline benchmark averaged 0.446 s per scan instead of 0.088 s.

## Synthetic scene

`lidar_interface/synthetic_scene.py:demo_scene` is the scene used for both runs below. It contains:

- a height-field terrain made of gentle waves plus eight mounds and craters;
- twelve rocks resting on the terrain, two walls, and a block;
- a sphere moving at 0.3 m/s in y;
- a sphere present only until t = 12 s;
- a box that appears at t = 10 s.

The sensor rides 0.6 m above the terrain. It drives 9.0 m in +x at 0.3 m/s and then stops. Each scan has 240 azimuth by 16 elevation rays over −25° to +15°, with 1 cm range noise.

## Offline trajectory benchmark

```bash
python3 /workspace/scripts/run_benchmark.py
```

This run used the YAML defaults from `config/`: 69 scans at 2 Hz, covering the full path plus 4 s stationary. Output files are `logs/benchmark_trajectory.jsonl` and `logs/benchmark_trajectory_summary.json`.

| Quantity | Value |
| --- | --- |
| Accepted / rejected | 64 / 5 (all `quality_gate`) |
| Final pose (x, y, z) | 8.722, −0.082, −0.124 m against truth 9.000, 0.000, 0.047 |
| Final / max / RMSE translation error | 0.337 / 0.452 / 0.225 m |
| Voxels (static) | 6666 (4010) |
| RSS | 255.3 MB |
| Points per scan in / after preprocessing | 2670 / 1184 |

Mean time per scan, in seconds:

| Stage | Mean time (s) |
| --- | --- |
| Synthetic generation | 0.164 |
| Preprocessing | 0.0034 |
| Features | 0.0079 |
| Candidates | 0.020 |
| CLIPPER | 0.016 |
| Registration (Kabsch, ICP, gate) | 0.011 |
| Map update | 0.035 |

The whole mapping session averaged 0.0876 s per scan, with a maximum of 0.163 s.

Change detection in the same run is measured as the number of map voxels inside each object's footprint above the terrain:

- **Box that appears at t = 10 s:** 0 voxels before it appears, 8 at the end.
- **Temporary sphere:** at most 13 voxels while it is present, 1 at the end.
- **Moving sphere's starting position:** at most 6 voxels, 0 at the end.

## End-to-end launch

```bash
bash /workspace/scripts/run_e2e.sh
python3 /workspace/scripts/summarize_run.py --since <launch start>
```

`run_e2e.sh` launches `pipeline.launch.py`, checks the visualizer after 15 s, stops the launch with SIGINT, and then summarizes the per-node metrics files `logs/metrics_*.jsonl`. The summary is saved to `logs/e2e_summary.json`. This run predates Foxglove: it requested the Viser page, which returned HTTP 200, and all metric lines decoded. The script now connects to `foxglove_bridge` instead, unless `VISUALIZER=viser`.

Throughput over the 40.5 s measurement window:

| Topic | Messages | Rate |
| --- | --- | --- |
| `/lidar/points` | 82 | 2.00 Hz |
| `/mapping/processed_scan` | 70 | 2.02 Hz |
| Accepted map updates | 63 (4 rejected, `quality_gate`) | 1.92 Hz |

The scans that were not processed are startup drops (the first 0 to 5.5 s, before the subscribers were up) and the last scan at shutdown.

Latency and resources:

| Quantity | Mean | p50 | p95 | Max |
| --- | --- | --- | --- | --- |
| Scan stamp to map published (s) | 0.415 | 0.383 | 0.513 | 1.150 |
| Mapping callback (s) | 0.214 | 0.210 | 0.290 | 0.338 |
| Preprocessing callback (s) | 0.0044 | 0.0040 | 0.0066 | 0.0119 |
| Synthetic generation (s) | 0.166 | 0.161 | 0.188 | 0.217 |
| Mapping RSS (MB) | 272.9 | 273.3 | 274.4 | 274.5 |

The scan stamp is taken at the start of the synthetic tick. As a result, the end-to-end latency includes about 0.17 s of ray casting that a real sensor would not add.

The mapping callback is slower than the offline session (0.214 s against 0.088 s). The callback also builds and publishes the global map, the registered scan, and the correspondence markers.

Tracking error, matched to ground truth by stamp:

- **Mean:** 0.271 m.
- **Max:** 0.398 m.
- **Final:** 0.397 m (estimated x 8.629 against truth 9.000), which is about 4% of the path.

## LAC stereo

```bash
python3 /workspace/scripts/lac_stereo_map.py "/workspace/software/pointclouds/example (2).lac"
```

This run used the 60 FrontLeft/FrontRight pairs from frames 20 to 1200. Each pair went through the same SGBM code as `ros2 run lidar_interface stereo_cloud`, at half resolution with depth limited to 0.5–6 m. Each cloud was placed in the world using the logged rover pose and the documented camera mount, then fused into a 0.1 m voxel map. No registration was involved. Output files are `logs/lac_stereo_summary.json`, `logs/lac_stereo.ply`, and `logs/lac_stereo.png`.

| Quantity | Value |
| --- | --- |
| Points per pair | 16098 mean |
| Stereo time per pair | 0.026 s |
| Map update per pair | 0.043 s |
| Median depth | 1.34 m |
| Median height of points below the camera | 0.26 m |
| Median distance from each frame's world points to the previous frame's | 0.013 m |
| Map | 9516 voxels, extent 13.0 × 21.6 × 2.35 m |

Between sampled frames, the rover moved a median of 0.30 m, for 17.5 m and 1.10 rad of yaw in total. Consecutive clouds still landing within about 1 cm means the focal length (1.22 rad horizontal field of view), the 0.162 m baseline, and the mount are consistent with the logged poses. This check does not measure absolute accuracy against a terrain model.

## Stereo pipeline on the LAC log

Both runs use the 60 front stereo pairs of `example (2).lac`, which are 1 s of mission time apart (frames 20 to 1200). The rover drives 17.5 m and turns up to 25 degrees between pairs. The pipeline is the LiDAR pipeline with the `config/stereo/` overrides. Error is measured against the logged rover pose, relative to the rover pose at the first pair. `OMP_NUM_THREADS=4`.

### Offline

```bash
python3 /workspace/scripts/run_stereo_benchmark.py
```

Output files are `logs/stereo_benchmark.jsonl` and `logs/stereo_benchmark_summary.json`.

| Quantity | Value |
| --- | --- |
| Accepted / rejected | 54 / 6 (5 `quality_gate`, 1 `too_few_candidates`) |
| Final translation error | 2.44 m (13.9% of 17.5 m) |
| RMSE of translation error | 1.26 m |
| Final heading error | −0.65 rad |
| Points per pair, stereo / after preprocessing | 16098 / 2161 |
| Map voxels at the end | 12842 |

Mean time per pair, in seconds:

| Stage | Mean time (s) |
| --- | --- |
| Stereo (SGBM, half resolution) | 0.034 |
| Preprocessing | 0.005 |
| Features | 0.031 |
| Candidates | 0.019 |
| CLIPPER (up to 1500 candidates) | 0.422 |
| Registration | 0.038 |
| Map update | 0.036 |
| Whole session | 0.527 |

### ROS launch

```bash
bash /workspace/scripts/run_stereo_e2e.sh
```

The launch log is `logs/stereo_e2e_launch.txt`, and the summary is `logs/stereo_e2e_summary.json`. Viser returned HTTP 200. All 60 pairs reached every stage: 60 stereo clouds, 60 preprocessed scans, and 54 accepted map updates. The 6 rejections were `quality_gate` and `too_few_candidates`.

| Quantity | Mean | p50 | p95 | Max |
| --- | --- | --- | --- | --- |
| Pair stamp to map published (s) | 0.895 | 0.790 | 1.855 | 2.167 |
| Mapping callback (s) | 0.677 | 0.636 | 0.989 | 1.325 |
| CLIPPER (s) | 0.417 | 0.386 | 0.711 | 1.122 |
| Stereo node (s) | 0.027 | 0.026 | 0.035 | 0.040 |
| Mapping RSS (MB) | 299.6 | 300.5 | 303.5 | 304.0 |
| Translation error (m) | 2.81 | 2.55 | 6.65 | 7.68 |

The final error in this run was 7.68 m, mostly in height: estimated z −6.06 m against −0.27 m. The offline run on the same pairs ended 2.44 m off. The only differences are that ROS carries the points as float32 and that the libraries are not repeatable. Either is enough to change which pairs are accepted, after which the drift takes a different path.

### Why stereo is harder than LiDAR here

These checks were made on the same pairs against the logged poses:

- **The true motion fits the data well.** Under the logged motion, consecutive clouds agree to a median 0.011 m point-to-plane residual with 88% of points within 0.1 m. The camera geometry and mount are consistent with the poses.
- **Stereo FPFH is not very distinctive.** Descriptor distances between true matches were 5 to 80, against a LiDAR cap of 2.0.
- **Grid keypoints bias toward no motion.** With voxel keypoints, about 8 of CLIPPER's 20 selected pairs were correct. With random keypoints, about 22 of 39 were.
- **The surface is nearly flat.** For the matched points, the smallest eigenvalue of the mean of n nᵀ over the surface normals was about 0.002 in most pairs, so forward and sideways slides are almost unconstrained. Point-to-plane ICP alone, started from no motion, had a median error of 0.06 to 0.14 m on a 0.3 m step.
- **Accepted steps carry a small rotation error.** The median per-step rotation error was 0.75 degrees. 14 to 16 of 53 steps exceeded 1.1 degrees, and one exceeded 18 degrees. A per-step rotation limit cannot separate that outlier, because the rover itself turns up to 25 degrees between pairs.
- **Image brightness would help.** Open3D's colored ICP, which uses image brightness as well as geometry, reached a median 0.04 to 0.05 m per step in a separate test. It is not part of the pipeline.

## Foxglove

Checked on 3 November 2026 with `foxglove_bridge` 3.6.0. `scripts/check_foxglove.py` subscribed through the bridge for 10 to 12 s and counted the messages delivered.

| Run | Messages received |
| --- | --- |
| `run_e2e.sh 45` (synthetic LiDAR) | `/viz/map` 19, `/viz/estimated_path` 21, `/viz/ground_truth_path` 20, `/mapping/registered_scan` 20, `/tf` 42 |
| `run_stereo_e2e.sh` (LAC stereo, from 30 s) | `/viz/map` 9, `/viz/estimated_path` 9, `/viz/ground_truth_path` 50, `/camera/left/image_raw` 10, `/lidar/points` 10, `/tf` 18 |
| `lac_view.launch.py loop:=true` (no mapping) | `/lidar/points` 12, `/camera/left/image_raw` 12, `/viz/ground_truth_path` 56, `/viz/ground_truth_pose` 238, `/tf` 237, `/tf_static` 1 |

With the bridge in place of Viser, the synthetic run ended 0.34 m off, and the stereo run accepted 55 of 60 pairs and ended 7.52 m off. Both are within the run-to-run variation below. The layout `config/foxglove/lunar_mapping.json` has not been loaded in the Foxglove app from this container, which has no display.

## Run-to-run variation

Open3D and CLIPPER do not give bit-identical results here. Two back-to-back runs of `scripts/run_benchmark.py` with the same configuration accepted 64 and 57 of 69 scans, with final errors of 0.352 m and 0.318 m. Treat single-run numbers in this file as one sample.

## Tests

The tests are run with `python3 -m pytest -q /workspace/ros2_ws/src`: 59 passed in 26.0 s. Since the stereo work, they also cover random keypoints, coasting after a rejection, placing the first scan at the sensor extrinsic, and reading the example LAC log. Earlier coverage:

- terrain and appearing-object scene tests;
- a CLIPPER test with 40 inliers at 2 cm noise plus 160 outliers (rotation error < 1°, translation error < 5 cm);
- a stationary-sensor map test where obstacles are removed, added, and moved;
- a session test that tracks the terrain drive with default settings;
- gated-candidate, extra-field `PointCloud2`, age-color, and stereo tests.

`python3 /opt/lunar/smoke_clipper.py` reported `clipperpy 0.2.4`, kept the six true inlier pairs, and printed `CLIPPER smoke test passed`.

The `lunar-mapping:humble` image build ran upstream CLIPPER `ctest`: 7 tests, 0 failed, 0.22 s. Eigen 3.3.9 was the CLIPPER dependency.

The Dockerfile now also installs `python3-opencv`, `ros-humble-cv-bridge`, `lunarloc`, and `ros-humble-foxglove-bridge`. Those packages were installed and imported in the running container. The image itself has not been rebuilt since that change.

## Unitree probe

`bash /workspace/scripts/probe_unitree.sh` on 2 November 2026, inside the dev container, not inside the image build. It installed `libpcl-dev` and `ros-humble-pcl-conversions`, cloned `unitreerobotics/unilidar_sdk2`, and `colcon build` of `unitree_lidar_ros2` finished in 30.3 s. CMake printed a CMP0074 warning about `PCL_ROOT` and did not fail. The log is `logs/unitree_probe.txt` and ends with `BUILD_OK`.

That package is not installed in `lunar-mapping:humble`. The generic `unitree_adapter` node is still only a `PointCloud2` remap, and this laptop has no L2 to publish `/unilidar/cloud`.
