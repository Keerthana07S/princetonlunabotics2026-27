"""Tests for range and field-of-view gates."""

from __future__ import annotations

import numpy as np

from lidar_interface.filters import SensorLimits, apply_limits, finite_mask
from lidar_interface.types import PointCloud


def test_finite_mask_drops_nonfinite_coordinates() -> None:
    points = np.array([[0.0, 0.0, 1.0], [np.nan, 1.0, 1.0], [1.0, np.inf, 1.0]])
    mask = finite_mask(points)
    assert mask.tolist() == [True, False, False]


def test_range_and_forward_fov() -> None:
    limits = SensorLimits(
        min_range=0.5,
        max_range=5.0,
        horizontal_fov_deg=90.0,
        horizontal_center_deg=0.0,
        vertical_min_deg=-20.0,
        vertical_max_deg=20.0,
    )
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [np.nan, 0.0, 1.0],
        ]
    )
    cloud = PointCloud(xyz=points, stamp_sec=1.0, frame_id="lidar_link")
    filtered, stats = apply_limits(cloud, limits)
    assert filtered.count == 1
    assert np.allclose(filtered.xyz[0], [2.0, 0.0, 0.0])
    assert stats["nonfinite"] == 1
    assert filtered.stamp_sec == 1.0
    assert filtered.frame_id == "lidar_link"
