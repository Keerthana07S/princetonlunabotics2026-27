"""Persistent spatial voxel hash map.

Unobserved space stays unknown. Free-space evidence applies only to voxels
that already exist, and only when the current scan saw past them: the scan is
binned into an azimuth/elevation range image, and a stored voxel is "seen
through" when every populated bin around its direction measured a range
beyond it by more than a margin. Requiring the whole neighbourhood keeps
grazing rays from erasing ground.
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
    free_margin: float = 0.3
    free_azimuth_deg: float = 2.0
    free_elevation_deg: float = 3.0


@dataclass
class MapCloud:
    """Exported geometry split by the static / dynamic label.

    ``last_observed`` is the stamp of the most recent hit on each voxel.
    """

    xyz: np.ndarray
    confidence: np.ndarray
    dynamic: np.ndarray
    last_observed: np.ndarray


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
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.shape[0] == 0:
            self.prune(stamp)
            return
        keys = np.floor(xyz / self.config.resolution).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, xyz)
        means = sums / np.bincount(inverse, minlength=unique_keys.shape[0])[:, None]
        hit_keys = {(int(k[0]), int(k[1]), int(k[2])) for k in unique_keys}
        self._apply_free_space(sensor, xyz, hit_keys, stamp)
        for key, mean in zip(unique_keys, means):
            self._observe(mean, stamp, (int(key[0]), int(key[1]), int(key[2])))
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
                last_observed=np.zeros((0,)),
            )
        voxels = list(self.voxels.values())
        return MapCloud(
            xyz=np.vstack([voxel.point for voxel in voxels]),
            confidence=np.asarray([voxel.confidence for voxel in voxels], dtype=np.float64),
            dynamic=np.asarray([voxel.dynamic for voxel in voxels], dtype=bool),
            last_observed=np.asarray([voxel.last_observed for voxel in voxels], dtype=np.float64),
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

    def _observe(self, point: np.ndarray, stamp: float, key: tuple[int, int, int] | None = None) -> None:
        if key is None:
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

    def _apply_free_space(
        self,
        origin: np.ndarray,
        points: np.ndarray,
        hit_keys: set[tuple[int, int, int]],
        stamp: float,
    ) -> None:
        if not self.voxels:
            return
        keys = [key for key in self.voxels if key not in hit_keys]
        if not keys:
            return
        stored = np.vstack([self.voxels[key].point for key in keys])
        az_step = math.radians(self.config.free_azimuth_deg)
        el_step = math.radians(self.config.free_elevation_deg)

        def bins(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            ranges = np.linalg.norm(offsets, axis=1)
            azimuth = np.arctan2(offsets[:, 1], offsets[:, 0])
            elevation = np.arcsin(np.clip(offsets[:, 2] / np.maximum(ranges, 1e-9), -1.0, 1.0))
            return (
                np.floor((azimuth + math.pi) / az_step).astype(np.int64),
                np.floor((elevation + 0.5 * math.pi) / el_step).astype(np.int64),
                ranges,
            )

        az_count = int(math.ceil(2.0 * math.pi / az_step))
        el_count = int(math.ceil(math.pi / el_step)) + 1
        hit_az, hit_el, hit_range = bins(points - origin)
        image = np.full((az_count, el_count + 2), np.inf, dtype=np.float64)
        np.minimum.at(image, (hit_az % az_count, hit_el + 1), hit_range)

        voxel_az, voxel_el, voxel_range = bins(stored - origin)
        required = voxel_range + self.config.free_margin + self.config.resolution
        seen_through = np.isfinite(image[voxel_az % az_count, voxel_el + 1])
        for d_az in (-1, 0, 1):
            for d_el in (-1, 0, 1):
                measured = image[(voxel_az + d_az) % az_count, voxel_el + 1 + d_el]
                seen_through &= ~np.isfinite(measured) | (measured > required)
        for row in np.flatnonzero(seen_through):
            voxel = self.voxels[keys[int(row)]]
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
