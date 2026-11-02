"""Stereo back-projection on a synthetic fronto-parallel plane."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lidar_interface.stereo import StereoConfig, disparity_to_points, focal_length, stereo_to_points  # noqa: E402


def _textured_pair(width: int, height: int, disparity: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    texture = rng.integers(0, 255, size=(height, width + disparity), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    texture = np.maximum(texture, 30)
    left = np.ascontiguousarray(texture[:, :width])
    right = np.ascontiguousarray(texture[:, disparity:])
    return left, right


def test_constant_disparity_gives_the_expected_depth() -> None:
    width, height, shift = 640, 360, 20
    left, right = _textured_pair(width, height, shift)
    config = StereoConfig(scale=1.0, num_disparities=64, block_size=5, min_depth=0.5, max_depth=20.0, stride=4)
    points = stereo_to_points(left, right, config)
    expected = focal_length(width, config.horizontal_fov_rad) * config.baseline / shift
    assert points.shape[0] > 1000
    assert np.median(points[:, 0]) == pytest.approx(expected, rel=0.03)
    assert points[:, 1].min() < -1.0 < 1.0 < points[:, 1].max()
    assert points[:, 2].min() < -0.5 < 0.5 < points[:, 2].max()


def test_back_projection_axes_follow_rep103() -> None:
    disparity = np.full((9, 9), np.nan, dtype=np.float32)
    disparity[0, 8] = 10.0
    config = StereoConfig(scale=1.0, stride=1, min_depth=0.1, max_depth=100.0)
    point = disparity_to_points(disparity, 9, config)[0]
    assert point[0] > 0.0
    assert point[1] < 0.0
    assert point[2] > 0.0


def test_dark_pixels_are_dropped() -> None:
    width, height, shift = 320, 180, 10
    left, right = _textured_pair(width, height, shift)
    left[:, : width // 2] = 0
    config = StereoConfig(scale=1.0, num_disparities=32, block_size=5, stride=2, max_depth=50.0)
    points = stereo_to_points(left, right, config)
    assert points.shape[0] > 0
    assert np.all(points[:, 1] < 0.2 * points[:, 0])
