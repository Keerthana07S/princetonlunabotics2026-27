"""Voxel aggregation, decay, free space, and stale removal."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dynamic_map.voxel_map import MapConfig, VoxelMap, traverse_voxels


def _config(**overrides: float | int) -> MapConfig:
    base = dict(
        resolution=1.0,
        observation_weight=0.4,
        decay_lambda=0.0,
        max_confidence=1.0,
        min_confidence=0.15,
        max_voxels=100,
        max_observation_count=20,
        min_static_observations=4,
        motion_threshold=0.4,
        contradiction_factor=0.5,
        free_weight=0.45,
        free_hits_to_clear=2,
    )
    base.update(overrides)
    return MapConfig(**base)


def test_repeated_observations_saturate_confidence() -> None:
    voxel_map = VoxelMap(_config(decay_lambda=0.0, observation_weight=0.4, max_confidence=1.0))
    point = np.array([[0.2, 0.2, 0.2]])
    for stamp in (0.0, 1.0, 2.0, 3.0):
        voxel_map.integrate_points(point, np.zeros(3), stamp)
    voxel = next(iter(voxel_map.voxels.values()))
    assert voxel.count == 4
    assert voxel.confidence == pytest.approx(1.0)
    assert voxel.dynamic is False


def test_observation_uses_the_exponential_decay_update() -> None:
    decay = math.log(2.0)
    voxel_map = VoxelMap(
        _config(decay_lambda=decay, observation_weight=0.25, max_confidence=5.0, min_static_observations=10)
    )
    point = np.array([[0.2, 0.2, 0.2]])
    voxel_map.integrate_points(point, np.zeros(3), 0.0)
    voxel_map.integrate_points(point, np.zeros(3), 1.0)
    voxel = next(iter(voxel_map.voxels.values()))
    expected = 0.25 * math.exp(-decay * 1.0) + 0.25
    assert voxel.confidence == pytest.approx(expected)


def test_decay_without_observation_halves_confidence() -> None:
    voxel_map = VoxelMap(_config(decay_lambda=math.log(2.0), observation_weight=1.0, max_confidence=5.0))
    voxel_map.integrate_points(np.array([[0.2, 0.2, 0.2]]), np.zeros(3), 0.0)
    voxel_map.decay_to(1.0)
    voxel = next(iter(voxel_map.voxels.values()))
    assert voxel.confidence == pytest.approx(0.5)


def test_ray_traversal_stops_before_the_hit_voxel() -> None:
    visited = traverse_voxels(np.array([0.1, 0.1, 0.1]), np.array([3.1, 0.1, 0.1]), 1.0)
    assert (3, 0, 0) not in visited
    assert (1, 0, 0) in visited
    assert (0, 0, 0) in visited


def test_free_space_removes_stale_geometry_and_ignores_unknown_space() -> None:
    voxel_map = VoxelMap(
        _config(
            observation_weight=0.5,
            free_weight=0.6,
            free_hits_to_clear=1,
            min_confidence=0.2,
            min_static_observations=10,
        )
    )
    voxel_map.integrate_points(np.array([[1.2, 0.1, 0.1]]), np.zeros(3), 0.0)
    assert len(voxel_map.voxels) == 1
    voxel_map.integrate_points(np.array([[3.6, 0.3, 0.3]]), np.zeros(3), 1.0)
    assert (1, 0, 0) not in voxel_map.voxels
    assert (3, 0, 0) in voxel_map.voxels
    assert len(voxel_map.voxels) == 1


def test_free_space_ignores_voxels_beside_the_ray() -> None:
    voxel_map = VoxelMap(_config(free_weight=0.6, free_hits_to_clear=1, min_static_observations=10))
    voxel_map.integrate_points(np.array([[1.2, 0.9, 0.1]]), np.zeros(3), 0.0)
    voxel_map.integrate_points(np.array([[3.6, 0.1, 0.1]]), np.zeros(3), 1.0)
    assert (1, 0, 0) in voxel_map.voxels


def test_grazing_ground_rays_do_not_erase_near_ground() -> None:
    voxel_map = VoxelMap(
        _config(
            resolution=0.2,
            free_weight=0.6,
            free_hits_to_clear=1,
            min_static_observations=10,
            decay_lambda=0.0,
            max_voxels=100_000,
        )
    )
    sensor = np.array([0.0, 0.0, 0.6])
    elevations = np.deg2rad(np.linspace(-25.0, -5.0, 9))
    azimuths = np.deg2rad(np.linspace(-20.0, 20.0, 21))
    elevation, azimuth = np.meshgrid(elevations, azimuths)
    distance = 0.6 / np.tan(-elevation)
    ground = np.stack(
        [distance * np.cos(azimuth), distance * np.sin(azimuth), np.zeros_like(distance)], axis=-1
    ).reshape(-1, 3)
    voxel_map.integrate_points(ground, sensor, 0.0)
    before = set(voxel_map.voxels)
    voxel_map.integrate_points(ground + np.array([0.05, 0.0, 0.0]), sensor + np.array([0.05, 0.0, 0.0]), 1.0)
    assert before <= set(voxel_map.voxels)


def test_moving_points_stay_out_of_the_static_export() -> None:
    voxel_map = VoxelMap(
        _config(min_static_observations=6, motion_threshold=0.3, resolution=0.5, decay_lambda=0.0)
    )
    origin = np.zeros(3)
    for stamp in range(3):
        voxel_map.integrate_points(np.array([[1.1, 0.1, 0.1]]), origin, float(stamp))
    assert voxel_map.static_points(3.0).shape[0] == 0
    for stamp in range(6):
        voxel_map.integrate_points(np.array([[0.2, 1.1, 0.1]]), origin, float(stamp))
    static = voxel_map.static_points(6.0)
    assert static.shape[0] == 1
    assert np.allclose(static[0, 1], 1.1, atol=0.2)


def test_low_confidence_voxels_are_removed() -> None:
    voxel_map = VoxelMap(_config(decay_lambda=math.log(2.0), observation_weight=0.2, min_confidence=0.15, max_confidence=1.0))
    voxel_map.integrate_points(np.array([[0.2, 0.2, 0.2]]), np.zeros(3), 0.0)
    voxel_map.decay_to(4.0)
    assert voxel_map.voxels == {}
