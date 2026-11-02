"""Shared geometric types for LiDAR scans."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointCloud:
    """A scan in one coordinate frame.

    ``xyz`` is an ``(N, 3)`` array in meters. ``stamp_sec`` is seconds and
    ``frame_id`` is the ROS frame the coordinates are expressed in.
    """

    xyz: np.ndarray
    stamp_sec: float
    frame_id: str
    intensity: np.ndarray | None = None
    label: np.ndarray | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        if self.intensity is not None:
            self.intensity = np.asarray(self.intensity, dtype=np.float64).reshape(-1)
        if self.label is not None:
            self.label = np.asarray(self.label, dtype=np.float64).reshape(-1)

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    def subset(self, mask: np.ndarray) -> "PointCloud":
        """Return the points selected by a boolean mask."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.count:
            raise ValueError("Mask length does not match the point cloud")
        intensity = None if self.intensity is None else self.intensity[mask]
        label = None if self.label is None else self.label[mask]
        extras = {name: values[mask] for name, values in self.extras.items()}
        return PointCloud(
            xyz=self.xyz[mask],
            stamp_sec=self.stamp_sec,
            frame_id=self.frame_id,
            intensity=intensity,
            label=label,
            extras=extras,
        )
