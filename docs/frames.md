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

Axes follow REP 103: x forward, y left, z up. Yaw is about +z. There is no GPS frame. The map frame is the LiDAR frame of the first accepted scan. This odometry drifts. Loop closure is not implemented.

`/sim/ground_truth_pose` is published by the synthetic driver only. The mapping node does not subscribe to it.
