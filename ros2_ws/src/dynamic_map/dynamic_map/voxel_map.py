"""Persistent spatial voxel hash map.

Unobserved space stays unknown. Free-space updates apply only to voxels a
sensor ray actually crosses, and only when that voxel already exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Voxel:
    """One occupied cell. Confidence is saturated and decays with time."""

    index: tuple[int, int, int]
    point: np.ndarray
    count: int
    confidence: float
    confidence_time: float
    last_observed: float
    last_free: float
    dynamic: bool
    stable_count: int
    free_count: int


@dataclass
class MapConfig:
    """Voxel resolution is meters. ``decay_lambda`` is 1/seconds.

    ``observation_weight`` and ``decay_lambda`` need tuning on a real sensor.
    """

    resolution: float = 0.2
    observation_weight: float = 0.35
    decay_lambda: float = 0.05
    max_confidence: float = 2.0
    min_confidence: float = 0.15
    max_voxels: int = 200_000
    max_observation_count: int = 40
    min_static_observations: int = 4
    motion_threshold: float = 0.15
    contradiction_factor: float = 0.5
    free_weight: float = 0.3
    free_hits_to_clear: int = 2


@dataclass
class MapCloud:
    """Exported geometry split by the static / dynamic label."""

    xyz: np.ndarray
    confidence: np.ndarray
    dynamic: np.ndarray


class VoxelMap:
    """Hash map of occupied voxels. The map does not grow by appending scans."""

    def __init__(self, config: MapConfig | None = None) -> None:
        self.config = config or MapConfig()
        if self.config.resolution <= 0.0:
            raise ValueError("voxel resolution must be positive")
        self.voxels: dict[tuple[int, int, int], Voxel] = {}

    def index_of(self, point: np.ndarray) -> tuple[int, int, int]:
        scaled = np.floor(np.asarray(point, dtype=np.float64) / self.config.resolution)
        return int(scaled[0]), int(scaled[1]), int(scaled[2])

    def decayed_confidence(self, voxel: Voxel, stamp: float) -> float:
        """``C_prev * exp(-lambda * dt)`` since the confidence was last written."""
        dt = max(0.0, float(stamp) - float(voxel.confidence_time))
        return float(voxel.confidence) * math.exp(-self.config.decay_lambda * dt)

    def _set_confidence(self, voxel: Voxel, value: float, stamp: float) -> None:
        voxel.confidence = min(max(0.0, value), self.config.max_confidence)
        voxel.confidence_time = float(stamp)

    def integrate_points(self, points: np.ndarray, origin: np.ndarray, stamp: float) -> None:
        """Fuse one registered scan expressed in the map frame.

        ``origin`` is the sensor position in that same frame.
        """
        xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        sensor = np.asarray(origin, dtype=np.float64).reshape(3)
        finite = np.isfinite(xyz).all(axis=1)
        for point in xyz[finite]:
            self._apply_free_space(sensor, point, stamp)
            self._observe(point, stamp)
        self.prune(stamp)

    def decay_to(self, stamp: float) -> None:
        """Apply temporal decay without adding an observation."""
        for voxel in self.voxels.values():
            self._set_confidence(voxel, self.decayed_confidence(voxel, stamp), stamp)
        self.prune(stamp)

    def prune(self, stamp: float) -> int:
        """Drop voxels below the confidence floor and enforce the memory cap."""
        removed = 0
        for key, voxel in list(self.voxels.items()):
            confidence = self.decayed_confidence(voxel, stamp)
            self._set_confidence(voxel, confidence, stamp)
            stale_dynamic = (
                voxel.dynamic
                and voxel.last_observed + 0.0 < stamp
                and (stamp - max(voxel.last_observed, voxel.last_free)) > self._dynamic_timeout()
            )
            if confidence < self.config.min_confidence or stale_dynamic:
                del self.voxels[key]
                removed += 1
        overflow = len(self.voxels) - self.config.max_voxels
        if overflow > 0:
            ordered = sorted(self.voxels.items(), key=lambda item: item[1].confidence)
            for key, _voxel in ordered[:overflow]:
                del self.voxels[key]
                removed += 1
        return removed

    def export(self, stamp: float) -> MapCloud:
        """Decay, prune, and return the remaining voxels."""
        self.prune(stamp)
        if not self.voxels:
            return MapCloud(
                xyz=np.zeros((0, 3)),
                confidence=np.zeros((0,)),
                dynamic=np.zeros((0,), dtype=bool),
            )
        points = []
        confidence = []
        dynamic = []
        for voxel in self.voxels.values():
            points.append(voxel.point)
            confidence.append(voxel.confidence)
            dynamic.append(voxel.dynamic)
        return MapCloud(
            xyz=np.vstack(points),
            confidence=np.asarray(confidence, dtype=np.float64),
            dynamic=np.asarray(dynamic, dtype=bool),
        )

    def static_points(self, stamp: float) -> np.ndarray:
        cloud = self.export(stamp)
        if cloud.xyz.shape[0] == 0:
            return cloud.xyz
        return cloud.xyz[~cloud.dynamic]

    def local_points(self, center: np.ndarray, radius: float, stamp: float, static_only: bool = True) -> np.ndarray:
        """Return map points inside a Euclidean radius of ``center``."""
        cloud = self.export(stamp)
        if cloud.xyz.shape[0] == 0:
            return cloud.xyz
        mask = np.linalg.norm(cloud.xyz - np.asarray(center, dtype=np.float64), axis=1) <= radius
        if static_only:
            mask &= ~cloud.dynamic
        return cloud.xyz[mask]

    def snapshot(self) -> dict[tuple[int, int, int], tuple[float, bool, int]]:
        """Confidence, dynamic flag, and count, for regression checks."""
        return {
            key: (voxel.confidence, voxel.dynamic, voxel.count)
            for key, voxel in self.voxels.items()
        }

    def _dynamic_timeout(self) -> float:
        if self.config.decay_lambda <= 0.0:
            return 5.0
        return max(1.0, 3.0 / self.config.decay_lambda)

    def _observe(self, point: np.ndarray, stamp: float) -> None:
        key = self.index_of(point)
        voxel = self.voxels.get(key)
        weight = self.config.observation_weight
        if voxel is None:
            created = Voxel(
                index=key,
                point=np.asarray(point, dtype=np.float64).copy(),
                count=1,
                confidence=0.0,
                confidence_time=stamp,
                last_observed=stamp,
                last_free=-1.0,
                dynamic=True,
                stable_count=1,
                free_count=0,
            )
            self._set_confidence(created, weight, stamp)
            self.voxels[key] = created
            return
        confidence = self.decayed_confidence(voxel, stamp) + weight
        displacement = float(np.linalg.norm(point - voxel.point))
        if displacement > self.config.motion_threshold:
            voxel.dynamic = True
            voxel.stable_count = 1
            self._set_confidence(voxel, confidence * self.config.contradiction_factor, stamp)
            voxel.point = np.asarray(point, dtype=np.float64).copy()
        else:
            count = min(voxel.count, self.config.max_observation_count)
            voxel.point = (voxel.point * count + point) / (count + 1)
            voxel.stable_count += 1
            self._set_confidence(voxel, confidence, stamp)
            if voxel.stable_count >= self.config.min_static_observations:
                voxel.dynamic = False
        voxel.count = min(voxel.count + 1, self.config.max_observation_count)
        voxel.last_observed = stamp
        voxel.free_count = 0

    def _apply_free_space(self, origin: np.ndarray, endpoint: np.ndarray, stamp: float) -> None:
        end_index = self.index_of(endpoint)
        for key in traverse_voxels(origin, endpoint, self.config.resolution):
            if key == end_index:
                continue
            voxel = self.voxels.get(key)
            if voxel is None:
                continue
            voxel.last_free = stamp
            voxel.free_count += 1
            updated = self.decayed_confidence(voxel, stamp) - self.config.free_weight
            if voxel.free_count >= self.config.free_hits_to_clear and stamp >= voxel.last_observed:
                updated = min(updated, self.config.min_confidence * 0.5)
                voxel.dynamic = True
            self._set_confidence(voxel, updated, stamp)


def traverse_voxels(start_pt: np.ndarray, end_pt: np.ndarray, resolution: float) -> list[tuple[int, int, int]]:
    """Visit voxels from ``start_pt`` up to but not including the end voxel.

    This is a 3D DDA traversal. It does not create an empty-space claim for
    voxels the caller does not already store.
    """
    start = np.floor(np.asarray(start_pt, dtype=np.float64) / resolution).astype(np.int64)
    end = np.floor(np.asarray(end_pt, dtype=np.float64) / resolution).astype(np.int64)
    if np.all(start == end):
        return []
    direction = np.asarray(end_pt, dtype=np.float64) - np.asarray(start_pt, dtype=np.float64)
    step = np.sign(direction).astype(np.int64)
    voxel = start.copy()
    t_max = np.full(3, np.inf, dtype=np.float64)
    t_delta = np.full(3, np.inf, dtype=np.float64)
    for axis in range(3):
        if step[axis] == 0:
            continue
        if step[axis] > 0:
            boundary = (voxel[axis] + 1) * resolution
        else:
            boundary = voxel[axis] * resolution
        t_max[axis] = (boundary - start_pt[axis]) / direction[axis]
        t_delta[axis] = resolution / abs(direction[axis])
    visited: list[tuple[int, int, int]] = []
    limit = int(np.abs(end - start).sum()) + 3
    for _ in range(limit):
        if np.all(voxel == end):
            break
        visited.append((int(voxel[0]), int(voxel[1]), int(voxel[2])))
        axis = int(np.argmin(t_max))
        if not np.isfinite(t_max[axis]):
            break
        voxel[axis] += step[axis]
        t_max[axis] += t_delta[axis]
    return visited
