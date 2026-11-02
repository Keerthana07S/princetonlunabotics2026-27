"""Add, remove, and move obstacles in a ray-cast scene and check the voxel map follows."""

from __future__ import annotations

import numpy as np

from dynamic_map.voxel_map import MapConfig, VoxelMap
from lidar_interface.synthetic_scene import Box, LidarModel, RoverPose, Scene, Sphere, Terrain

CHANGE_TIME = 5.0


def _scene() -> Scene:
    return Scene(
        [
            Terrain("ground", wave_amplitude=0.0),
            # Free-space evidence needs a return behind the object; rays into the sky prove nothing.
            Box("wall", np.array([7.0, -8.0, -0.5]), np.array([7.3, 8.0, 3.0])),
            Sphere("removed", np.array([3.0, -1.5, 0.5]), 0.5, active_end=CHANGE_TIME - 0.5),
            Box("added", np.array([3.0, 1.0, 0.0]), np.array([3.6, 1.6, 1.0]), active_start=CHANGE_TIME),
            Sphere("moved_before", np.array([4.0, 0.0, 0.5]), 0.5, active_end=CHANGE_TIME - 0.5),
            Sphere("moved_after", np.array([4.0, -3.0, 0.5]), 0.5, active_start=CHANGE_TIME),
        ],
        sensor_height=0.6,
    )


def _object_voxels(voxel_map: VoxelMap, primitive: Sphere | Box, stamp: float) -> int:
    points = voxel_map.export(stamp).xyz
    if points.shape[0] == 0:
        return 0
    if isinstance(primitive, Sphere):
        near = np.linalg.norm(points - primitive.center, axis=1) <= primitive.radius + 0.15
    else:
        near = np.all((points >= primitive.minimum - 0.15) & (points <= primitive.maximum + 0.15), axis=1)
    return int(np.sum(near & (points[:, 2] > 0.35)))


def _ground_voxels(voxel_map: VoxelMap, stamp: float) -> int:
    points = voxel_map.export(stamp).xyz
    return int(np.sum(np.abs(points[:, 2]) < 0.1))


def test_map_adds_removes_and_moves_obstacles() -> None:
    scene = _scene()
    objects = {primitive.name: primitive for primitive in scene.primitives if not isinstance(primitive, Terrain)}
    pose = scene.sensor_pose(RoverPose(position=np.zeros(3), yaw=0.0))
    model = LidarModel(vertical_min_deg=-25.0, vertical_max_deg=15.0, azimuth_bins=240, elevation_bins=16, seed=4)
    voxel_map = VoxelMap(MapConfig(resolution=0.2, min_static_observations=4, decay_lambda=0.05))

    def integrate(stamp: float) -> None:
        local = scene.scan(pose, stamp, model)
        voxel_map.integrate_points(local + pose.position, pose.position, stamp)

    for stamp in np.arange(0.0, CHANGE_TIME - 0.5, 0.5):
        integrate(float(stamp))
    before_stamp = CHANGE_TIME - 1.0
    removed_before = _object_voxels(voxel_map, objects["removed"], before_stamp)
    moved_before = _object_voxels(voxel_map, objects["moved_before"], before_stamp)
    ground_before = _ground_voxels(voxel_map, before_stamp)
    assert removed_before >= 10
    assert moved_before >= 10
    assert _object_voxels(voxel_map, objects["added"], before_stamp) == 0
    assert _object_voxels(voxel_map, objects["moved_after"], before_stamp) == 0
    static = voxel_map.export(before_stamp)
    assert np.mean(~static.dynamic) > 0.5

    for stamp in np.arange(CHANGE_TIME, CHANGE_TIME + 4.0, 0.5):
        integrate(float(stamp))
    after_stamp = CHANGE_TIME + 3.5
    assert _object_voxels(voxel_map, objects["removed"], after_stamp) <= removed_before // 10
    assert _object_voxels(voxel_map, objects["moved_before"], after_stamp) <= moved_before // 10
    assert _object_voxels(voxel_map, objects["added"], after_stamp) >= 10
    assert _object_voxels(voxel_map, objects["moved_after"], after_stamp) >= 10
    assert _ground_voxels(voxel_map, after_stamp) >= 0.9 * ground_before
