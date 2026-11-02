"""Sensor-frame limits that do not depend on a LiDAR vendor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_interface.types import PointCloud


@dataclass
class SensorLimits:
    """Range and field-of-view gate applied in the sensor frame.

    Horizontal angles use atan2(y, x): 0 is forward, positive is left.
    A horizontal field of view of 360 degrees or more keeps every azimuth.
    Vertical angles are elevation in degrees.
    """

    min_range: float = 0.2
    max_range: float = 30.0
    horizontal_fov_deg: float = 360.0
    horizontal_center_deg: float = 0.0
    vertical_min_deg: float = -15.0
    vertical_max_deg: float = 45.0

    def __post_init__(self) -> None:
        if self.max_range <= self.min_range:
            raise ValueError("max_range must be greater than min_range")
        if self.vertical_max_deg < self.vertical_min_deg:
            raise ValueError("vertical FOV limits are reversed")


def finite_mask(xyz: np.ndarray) -> np.ndarray:
    """True where x, y, and z are all finite."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    return np.isfinite(points).all(axis=1)


def range_mask(xyz: np.ndarray, limits: SensorLimits) -> np.ndarray:
    """True for points whose range lies inside the configured interval."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    ranges = np.linalg.norm(points, axis=1)
    return (ranges >= limits.min_range) & (ranges <= limits.max_range)


def _wrap_degrees(angle: np.ndarray) -> np.ndarray:
    return (angle + 180.0) % 360.0 - 180.0


def fov_mask(xyz: np.ndarray, limits: SensorLimits) -> np.ndarray:
    """True for points inside the horizontal and vertical field of view."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    horizontal = np.zeros(points.shape[0], dtype=bool)
    if limits.horizontal_fov_deg >= 360.0:
        horizontal[:] = True
    else:
        azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        delta = np.abs(_wrap_degrees(azimuth - limits.horizontal_center_deg))
        horizontal = delta <= (limits.horizontal_fov_deg * 0.5)
    planar = np.linalg.norm(points[:, :2], axis=1)
    elevation = np.degrees(np.arctan2(points[:, 2], planar))
    vertical = (elevation >= limits.vertical_min_deg) & (elevation <= limits.vertical_max_deg)
    return horizontal & vertical


def apply_limits(cloud: PointCloud, limits: SensorLimits) -> tuple[PointCloud, dict[str, int]]:
    """Drop non-finite points and points outside range or field of view."""
    finite = finite_mask(cloud.xyz)
    ranged = range_mask(cloud.xyz, limits)
    visible = fov_mask(cloud.xyz, limits)
    keep = finite & ranged & visible
    stats = {
        "input": cloud.count,
        "nonfinite": int((~finite).sum()),
        "out_of_range": int((finite & ~ranged).sum()),
        "out_of_fov": int((finite & ranged & ~visible).sum()),
        "output": int(keep.sum()),
    }
    return cloud.subset(keep), stats
