"""Scan-to-scan and scan-to-submap mapping session.

This is LiDAR odometry plus a voxel map. It does not close loops.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from dynamic_map.voxel_map import MapConfig, VoxelMap
from lidar_interface.geometry import transform_points
from pointcloud_processing.clipper_association import AssociationResult, ClipperConfig, associate
from pointcloud_processing.features import FeatureConfig, FeatureSet, candidate_correspondences, extract_features
from pointcloud_processing.registration import RegistrationConfig, register_correspondences

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Registration mode is ``scan_to_scan`` or ``scan_to_submap``."""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    clipper: ClipperConfig = field(default_factory=ClipperConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    voxel: MapConfig = field(default_factory=MapConfig)
    mode: str = "scan_to_scan"
    submap_radius: float = 8.0
    min_points: int = 30
    motion_prediction: bool = True


@dataclass
class ScanResult:
    """Outcome of one scan. ``transform`` is ``T_map_lidar`` after this scan."""

    accepted: bool
    reason: str
    transform: np.ndarray
    rmse: float
    fitness: float
    inliers: int
    timings: dict[str, float]
    correspondence_lines: np.ndarray
    overlap: float = 0.0


class MappingSession:
    """Incrementally register scans. A rejected scan leaves the map unchanged."""

    def __init__(self, config: SessionConfig | None = None) -> None:
        self.config = config or SessionConfig()
        if self.config.mode not in ("scan_to_scan", "scan_to_submap"):
            raise ValueError("mode must be scan_to_scan or scan_to_submap")
        self.map = VoxelMap(self.config.voxel)
        self.transform = np.eye(4)
        self._previous_points: np.ndarray | None = None
        self._previous_features: FeatureSet | None = None
        self._previous_stamp = 0.0
        self._velocity: tuple[np.ndarray, float] | None = None
        self.initialized = False

    def predicted_motion(self, stamp: float) -> np.ndarray:
        """Constant-velocity guess of ``T_prev_curr`` from the last accepted step."""
        if not self.config.motion_prediction or self._velocity is None:
            return np.eye(4)
        relative, step_dt = self._velocity
        dt = float(stamp) - self._previous_stamp
        if step_dt <= 1e-6 or dt <= 0.0:
            return np.eye(4)
        scale = min(dt / step_dt, 30.0)
        predicted = np.eye(4)
        predicted[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(relative[:3, :3]).as_rotvec() * scale).as_matrix()
        predicted[:3, 3] = relative[:3, 3] * scale
        return predicted

    def process_scan(self, points: np.ndarray, stamp: float) -> ScanResult:
        """Register ``points`` in the sensor frame and integrate them when accepted."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.shape[0] < self.config.min_points:
            return self._reject("too_few_points", started, timings)
        feature_started = time.perf_counter()
        features = extract_features(xyz, self.config.features, sensor_origin=np.zeros(3))
        timings["features_s"] = time.perf_counter() - feature_started
        if features.count < self.config.clipper.min_inliers:
            return self._reject("too_few_features", started, timings)

        if not self.initialized:
            self._integrate(xyz, np.eye(4), stamp)
            self._previous_points = xyz
            self._previous_features = features
            self._previous_stamp = float(stamp)
            self.initialized = True
            timings["total_s"] = time.perf_counter() - started
            return ScanResult(
                accepted=True,
                reason="initialized",
                transform=self.transform.copy(),
                rmse=0.0,
                fitness=1.0,
                inliers=features.count,
                timings=timings,
                correspondence_lines=np.zeros((0, 2, 3)),
            )

        target_points, target_features, target_in_map = self._target(stamp)
        if target_features is None or target_features.count < self.config.clipper.min_inliers:
            return self._reject("no_target_features", started, timings)
        pair_started = time.perf_counter()
        motion = self.predicted_motion(stamp)
        predicted = self.transform @ motion if target_in_map else motion
        candidates = candidate_correspondences(features, target_features, self.config.features, predicted)
        timings["candidates_s"] = time.perf_counter() - pair_started
        clipper_started = time.perf_counter()
        association = associate(features.xyz, target_features.xyz, candidates, self.config.clipper)
        timings["clipper_s"] = time.perf_counter() - clipper_started
        if association.diagnostics.get("reason") != "ok":
            return self._reject(str(association.diagnostics.get("reason")), started, timings, association)

        register_started = time.perf_counter()
        registration = register_correspondences(
            features.xyz,
            target_features.xyz,
            association.inliers,
            self.config.registration,
            source_cloud=xyz,
            target_cloud=target_points,
        )
        timings["registration_s"] = time.perf_counter() - register_started
        if not registration.accepted:
            rejected = self._reject(
                registration.reason, started, timings, association, registration.rmse, registration.fitness
            )
            rejected.overlap = registration.overlap
            return rejected

        if self.config.mode == "scan_to_scan":
            updated = self.transform @ registration.transform
        else:
            updated = registration.transform
        map_started = time.perf_counter()
        self._integrate(xyz, updated, stamp)
        timings["map_s"] = time.perf_counter() - map_started
        lines = self._lines(features.xyz, target_features.xyz, association, updated, target_in_map)
        self._velocity = (np.linalg.inv(self.transform) @ updated, float(stamp) - self._previous_stamp)
        self.transform = updated
        self._previous_points = xyz
        self._previous_features = features
        self._previous_stamp = float(stamp)
        timings["total_s"] = time.perf_counter() - started
        return ScanResult(
            accepted=True,
            reason="accepted",
            transform=updated.copy(),
            rmse=registration.rmse,
            fitness=registration.fitness,
            inliers=registration.inlier_count,
            timings=timings,
            correspondence_lines=lines,
            overlap=registration.overlap,
        )

    def _target(self, stamp: float) -> tuple[np.ndarray, FeatureSet | None, bool]:
        if self.config.mode == "scan_to_submap":
            center = self.transform[:3, 3]
            local = self.map.local_points(center, self.config.submap_radius, stamp, static_only=True)
            if local.shape[0] < self.config.min_points:
                local = self.map.local_points(center, self.config.submap_radius, stamp, static_only=False)
            features = extract_features(local, self.config.features, sensor_origin=center)
            return local, features, True
        assert self._previous_points is not None and self._previous_features is not None
        return self._previous_points, self._previous_features, False

    def _integrate(self, points: np.ndarray, transform: np.ndarray, stamp: float) -> None:
        mapped = transform_points(points, transform)
        self.map.integrate_points(mapped, transform[:3, 3], stamp)

    def _lines(
        self,
        source: np.ndarray,
        target: np.ndarray,
        association: AssociationResult,
        updated: np.ndarray,
        target_in_map: bool,
    ) -> np.ndarray:
        if association.count == 0:
            return np.zeros((0, 2, 3))
        source_map = transform_points(source[association.inliers[:, 0]], updated)
        if target_in_map:
            target_map = target[association.inliers[:, 1]]
        else:
            target_map = transform_points(target[association.inliers[:, 1]], self.transform)
        return np.stack([source_map, target_map], axis=1)

    def _reject(
        self,
        reason: str,
        started: float,
        timings: dict[str, float],
        association: AssociationResult | None = None,
        rmse: float = float("inf"),
        fitness: float = 0.0,
    ) -> ScanResult:
        timings["total_s"] = time.perf_counter() - started
        logger.info("Registration rejected (%s); the existing map was kept", reason)
        return ScanResult(
            accepted=False,
            reason=reason,
            transform=self.transform.copy(),
            rmse=rmse,
            fitness=fitness,
            inliers=0 if association is None else association.count,
            timings=timings,
            correspondence_lines=np.zeros((0, 2, 3)),
        )
