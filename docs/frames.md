# Coordinate frames

`T_map_lidar` is a 4x4 matrix. A point in the LiDAR frame becomes a point in the map by

```text
p_map = T_map_lidar * p_lidar
```

`T_target_source` maps a source scan into the target scan. For scan-to-scan registration the target is the previous LiDAR frame and the source is the current LiDAR frame, so

```text
T_map_current = T_map_previous * T_previous_current
```

For scan-to-submap registration the target is already the map, and `T_target_source` is `T_map_lidar`.

The rover pose on `/mapping/rover_pose` is `T_map_base`. With the default extrinsic, the LiDAR frame and `base_link` are the same. A non-zero extrinsic uses

```text
T_map_base = T_map_lidar * inverse(T_base_lidar)
```

Axes follow REP 103: x forward, y left, z up. Yaw is about +z. There is no GPS frame. The map frame is `base_link` at the first accepted scan, so that scan is integrated at `T_map_lidar = T_base_lidar` (the `lidar_x/y/z/yaw` extrinsic). With the default zero extrinsic this is also the first scan's LiDAR frame. This odometry drifts. Loop closure is not implemented.

For stereo the "LiDAR" frame is the left camera's body frame (x forward along the optical axis, y left, z up), and the extrinsic is the camera mount. The LAC front-left camera is at (0.280, 0.081, 0.131) m in the rover frame with no rotation, as logged in the `.lac` file.

`/sim/ground_truth_pose` is published by the synthetic driver only. The mapping node does not subscribe to it.
