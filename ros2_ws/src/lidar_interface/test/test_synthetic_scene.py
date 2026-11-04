"""Tests for the ray-cast scene."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_interface.synthetic_scene import (
    Box,
    LidarModel,
    Mound,
    RoverPose,
    Scene,
    Sphere,
    Terrain,
    demo_scene,
    line_pose,
)


def _dense_model(**overrides: float) -> LidarModel:
    values = dict(
        min_range=0.1,
        max_range=20.0,
        horizontal_fov_deg=40.0,
        horizontal_center_deg=0.0,
        vertical_min_deg=-10.0,
        vertical_max_deg=20.0,
        azimuth_bins=21,
        elevation_bins=11,
        range_noise_std=0.0,
        seed=1,
    )
    values.update(overrides)
    return LidarModel(**values)


def test_wall_returns_points_near_its_face() -> None:
    scene = Scene([Box("wall", np.array([4.0, -1.0, 0.0]), np.array([4.2, 1.0, 2.0]))])
    pose = RoverPose(position=np.zeros(3), yaw=0.0)
    cloud = scene.scan(pose, 0.0, _dense_model())
    assert cloud.shape[0] > 20
    assert np.all(cloud[:, 0] > 3.8)
    assert np.all(cloud[:, 0] < 4.4)


def test_max_range_drops_a_distant_wall() -> None:
    scene = Scene([Box("wall", np.array([8.0, -1.0, 0.0]), np.array([8.2, 1.0, 2.0]))])
    pose = RoverPose(position=np.zeros(3), yaw=0.0)
    cloud = scene.scan(pose, 0.0, _dense_model(max_range=5.0))
    assert cloud.shape[0] == 0


def test_near_sphere_occludes_the_wall() -> None:
    scene = Scene(
        [
            Sphere("blocker", np.array([2.0, 0.0, 0.4]), 0.4),
            Box("wall", np.array([6.0, -1.0, 0.0]), np.array([6.2, 1.0, 2.0])),
        ]
    )
    pose = RoverPose(position=np.zeros(3), yaw=0.0)
    cloud = scene.scan(pose, 0.0, _dense_model(azimuth_bins=5, elevation_bins=5, horizontal_fov_deg=8.0))
    assert cloud.shape[0] > 0
    assert np.median(cloud[:, 0]) < 3.0


def test_moving_sphere_changes_hit_location() -> None:
    scene = Scene([Sphere("mover", np.array([3.0, -1.0, 0.4]), 0.35, velocity=np.array([0.0, 1.0, 0.0]))])
    pose = RoverPose(position=np.zeros(3), yaw=0.0)
    model = _dense_model(horizontal_fov_deg=80.0, azimuth_bins=41)
    early = scene.scan(pose, 0.0, model)
    late = scene.scan(pose, 1.5, model)
    assert early.shape[0] > 0 and late.shape[0] > 0
    assert np.mean(late[:, 1]) > np.mean(early[:, 1]) + 0.5


def test_object_outside_its_time_window_is_absent() -> None:
    scene = Scene(
        [Sphere("temp", np.array([3.0, 0.0, 0.4]), 0.4, active_start=0.0, active_end=1.0)]
    )
    pose = RoverPose(position=np.zeros(3), yaw=0.0)
    model = _dense_model()
    assert scene.scan(pose, 0.5, model).shape[0] > 0
    assert scene.scan(pose, 2.0, model).shape[0] == 0


def test_line_motion_advances_along_x() -> None:
    start = line_pose(0.0, speed=0.4)
    later = line_pose(2.0, speed=0.4)
    assert later.position[0] == pytest.approx(0.8)
    assert start.position[0] == 0.0


def test_line_motion_stops_at_max_distance() -> None:
    assert line_pose(100.0, speed=0.4, max_distance=3.0).position[0] == pytest.approx(3.0)


def test_object_appears_after_its_start_time() -> None:
    scene = Scene([Box("arrival", np.array([3.0, -0.5, 0.0]), np.array([3.4, 0.5, 1.0]), active_start=5.0)])
    pose = RoverPose(position=np.array([0.0, 0.0, 0.4]), yaw=0.0)
    model = _dense_model()
    assert scene.scan(pose, 4.0, model).shape[0] == 0
    assert scene.scan(pose, 6.0, model).shape[0] > 0


def test_flat_terrain_hits_lie_on_the_ground_plane() -> None:
    terrain = Terrain("flat", wave_amplitude=0.0)
    scene = Scene([terrain], sensor_height=0.8)
    pose = scene.sensor_pose(RoverPose(position=np.zeros(3), yaw=0.0))
    assert pose.position[2] == pytest.approx(0.8)
    cloud = scene.scan(pose, 0.0, _dense_model(vertical_min_deg=-30.0, vertical_max_deg=-5.0))
    assert cloud.shape[0] > 50
    np.testing.assert_allclose(cloud[:, 2], -0.8, atol=0.01)


def test_terrain_mound_raises_ground_returns() -> None:
    model = _dense_model(vertical_min_deg=-30.0, vertical_max_deg=-5.0, horizontal_fov_deg=10.0)
    flat = Scene([Terrain("flat", wave_amplitude=0.0)], sensor_height=0.8)
    hilly = Scene([Terrain("hill", wave_amplitude=0.0, mounds=[Mound((3.0, 0.0), 0.5, 0.8)])], sensor_height=0.8)
    pose = RoverPose(position=np.zeros(3), yaw=0.0)
    flat_cloud = flat.scan(flat.sensor_pose(pose), 0.0, model)
    hilly_cloud = hilly.scan(hilly.sensor_pose(pose), 0.0, model)
    assert hilly_cloud.shape[0] > 0
    assert np.max(hilly_cloud[:, 2]) > np.max(flat_cloud[:, 2]) + 0.3


def test_sensor_rides_on_terrain_height() -> None:
    terrain = Terrain("hill", wave_amplitude=0.0, mounds=[Mound((2.0, 0.0), 0.5, 0.8)])
    scene = Scene([terrain], sensor_height=0.6)
    on_hill = scene.sensor_pose(RoverPose(position=np.array([2.0, 0.0, 0.0]), yaw=0.0))
    assert on_hill.position[2] == pytest.approx(1.1)


def test_demo_scene_has_dense_ground_and_obstacles() -> None:
    scene = demo_scene()
    pose = scene.sensor_pose(line_pose(0.0, speed=0.3))
    cloud = scene.scan(pose, 0.0, LidarModel(vertical_min_deg=-25.0, vertical_max_deg=15.0, azimuth_bins=240))
    assert cloud.shape[0] > 1500
    assert np.any(cloud[:, 2] > 0.0)
