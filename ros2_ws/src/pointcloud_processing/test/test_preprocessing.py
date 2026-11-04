"""Preprocessing tests."""

from __future__ import annotations

import numpy as np

from lidar_interface.geometry import make_transform, yaw_rotation
from pointcloud_processing.preprocessing import (
    PreprocessConfig,
    preprocess,
    radius_outlier_removal,
    segment_clusters,
    statistical_outlier_removal,
    voxel_downsample,
)


def test_voxel_downsample_reduces_a_dense_line() -> None:
    line = np.stack([np.linspace(0.0, 2.0, 400), np.zeros(400), np.zeros(400)], axis=1)
    down = voxel_downsample(line, 0.2)
    assert 5 <= down.shape[0] < 40


def test_statistical_outlier_removes_a_distant_point() -> None:
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(80, 3))
    points = np.vstack([cluster, np.array([[50.0, 50.0, 50.0]])])
    cleaned = statistical_outlier_removal(points, neighbors=10, std_ratio=1.0)
    assert cleaned.shape[0] < points.shape[0]
    assert np.max(np.linalg.norm(cleaned, axis=1)) < 5.0


def test_radius_outlier_removes_an_isolated_point() -> None:
    cluster = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.05, 0.05, 0.0]])
    points = np.vstack([cluster, np.array([[3.0, 3.0, 3.0]])])
    cleaned = radius_outlier_removal(points, radius=0.2, min_neighbors=2)
    assert cleaned.shape[0] == 4


def test_clusters_split_two_groups() -> None:
    first = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]])
    second = first + np.array([2.0, 0.0, 0.0])
    clusters = segment_clusters(np.vstack([first, second]), eps=0.2, min_points=3)
    assert len(clusters) == 2


def test_preprocess_drops_far_points_and_keeps_stamp_contract() -> None:
    near = np.array([[1.0, 0.0, 0.0], [1.1, 0.05, 0.0], [0.9, -0.05, 0.02], [1.0, 0.1, -0.02]])
    far = np.array([[100.0, 0.0, 0.0]])
    config = PreprocessConfig(
        min_range=0.2,
        max_range=10.0,
        voxel_size=0.0,
        statistical_enabled=False,
        radius_enabled=False,
    )
    result = preprocess(np.vstack([near, far]), config)
    assert result.shape[0] == 4
    assert np.max(result[:, 0]) < 5.0


def test_transform_matches_yaw() -> None:
    from pointcloud_processing.preprocessing import transform_cloud

    points = np.array([[1.0, 0.0, 0.0]])
    transform = make_transform(yaw_rotation(np.pi / 2.0), np.zeros(3))
    moved = transform_cloud(points, transform)
    assert np.allclose(moved, [[0.0, 1.0, 0.0]], atol=1e-8)
