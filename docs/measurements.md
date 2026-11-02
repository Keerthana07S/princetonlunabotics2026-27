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

`run_e2e.sh` launches `pipeline.launch.py`, requests the Viser page after 15 s, stops the launch with SIGINT, and then summarizes the per-node metrics files `logs/metrics_*.jsonl`. The summary is saved to `logs/e2e_summary.json`. In this run, Viser returned HTTP 200 and all metric lines decoded.

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

## Tests

The tests are run with `python3 -m pytest -q /workspace/ros2_ws/src`: 53 passed in 13.3 s. The new coverage is:

- terrain and appearing-object scene tests;
- a CLIPPER test with 40 inliers at 2 cm noise plus 160 outliers (rotation error < 1°, translation error < 5 cm);
- a stationary-sensor map test where obstacles are removed, added, and moved;
- a session test that tracks the terrain drive with default settings;
- gated-candidate, extra-field `PointCloud2`, age-color, and stereo tests.

`python3 /opt/lunar/smoke_clipper.py` reported `clipperpy 0.2.4`, kept the six true inlier pairs, and printed `CLIPPER smoke test passed`.

The `lunar-mapping:humble` image build ran upstream CLIPPER `ctest`: 7 tests, 0 failed, 0.22 s. Eigen 3.3.9 was the CLIPPER dependency.

The Dockerfile now also installs `python3-opencv`, `ros-humble-cv-bridge`, and `lunarloc`. Those packages were installed and imported in the running container. The image itself has not been rebuilt since that change.

## Unitree probe

`bash /workspace/scripts/probe_unitree.sh` on 2 November 2026, inside the dev container, not inside the image build. It installed `libpcl-dev` and `ros-humble-pcl-conversions`, cloned `unitreerobotics/unilidar_sdk2`, and `colcon build` of `unitree_lidar_ros2` finished in 30.3 s. CMake printed a CMP0074 warning about `PCL_ROOT` and did not fail. The log is `logs/unitree_probe.txt` and ends with `BUILD_OK`.

That package is not installed in `lunar-mapping:humble`. The generic `unitree_adapter` node is still only a `PointCloud2` remap, and this laptop has no L2 to publish `/unilidar/cloud`.
