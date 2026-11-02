"""Voxel keypoints and FPFH descriptors. These propose correspondences; CLIPPER selects them."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pointcloud_processing.preprocessing import voxel_downsample

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Keypoint and descriptor settings. Radii are in meters and need sensor tuning."""

    keypoint_voxel: float = 0.3
    normal_radius: float = 0.6
    normal_max_nn: int = 30
    fpfh_radius: float = 0.9
    fpfh_max_nn: int = 100
    max_correspondences: int = 80
    max_descriptor_distance: float = 1.5
    mutual: bool = True
    candidates_per_keypoint: int = 1
    gate_radius: float = 0.0


@dataclass
class FeatureSet:
    """Keypoint coordinates ``(K, 3)`` and FPFH rows ``(K, 33)``."""

    xyz: np.ndarray
    descriptors: np.ndarray

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])


def _empty() -> FeatureSet:
    return FeatureSet(xyz=np.zeros((0, 3)), descriptors=np.zeros((0, 33)))


def extract_features(points: np.ndarray, config: FeatureConfig, sensor_origin: np.ndarray | None = None) -> FeatureSet:
    """Downsample to keypoints and compute FPFH. Fewer than 8 points returns an empty set."""
    keypoints = voxel_downsample(points, config.keypoint_voxel)
    if keypoints.shape[0] < 8:
        logger.info("Not enough points (%d) for FPFH", keypoints.shape[0])
        return _empty()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(keypoints)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(config.normal_radius), max_nn=int(config.normal_max_nn))
    )
    origin = np.zeros(3) if sensor_origin is None else np.asarray(sensor_origin, dtype=np.float64)
    cloud.orient_normals_towards_camera_location(origin)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(config.fpfh_radius), max_nn=int(config.fpfh_max_nn)),
    )
    if fpfh is None:
        logger.warning("FPFH returned no descriptors")
        return _empty()
    descriptors = np.asarray(fpfh.data, dtype=np.float64).T
    if descriptors.shape != (keypoints.shape[0], 33):
        logger.warning("Unexpected FPFH shape %s", descriptors.shape)
        return _empty()
    return FeatureSet(xyz=keypoints, descriptors=descriptors)


def candidate_correspondences(
    source: FeatureSet,
    target: FeatureSet,
    config: FeatureConfig,
    predicted: np.ndarray | None = None,
) -> np.ndarray:
    """Putative pairs from FPFH similarity, capped at ``max_correspondences``.

    With ``predicted`` (a ``T_target_source`` guess) and a positive
    ``gate_radius``, each source keypoint is matched only against target
    keypoints within ``gate_radius`` of its predicted position, keeping the
    ``candidates_per_keypoint`` closest descriptors. Otherwise this is
    (optionally mutual) nearest neighbour matching in FPFH space.

    The result is an ``(K, 2)`` int array of ``(source_index, target_index)``.
    This is a putative set. It is not the CLIPPER inlier decision.
    """
    if source.count == 0 or target.count == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if predicted is not None and config.gate_radius > 0.0:
        return _gated_candidates(source, target, config, np.asarray(predicted, dtype=np.float64))
    target_tree = cKDTree(target.descriptors)
    distances, indices = target_tree.query(source.descriptors, k=1)
    distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
    indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
    reverse_index = None
    if config.mutual:
        source_tree = cKDTree(source.descriptors)
        _reverse_distance, reverse_index = source_tree.query(target.descriptors, k=1)
        reverse_index = np.atleast_1d(np.asarray(reverse_index, dtype=np.int64))

    pairs: list[tuple[int, int]] = []
    pair_distance: list[float] = []
    for source_index, (target_index, distance) in enumerate(zip(indices, distances)):
        if target_index < 0 or target_index >= target.count:
            continue
        if not np.isfinite(distance) or distance > config.max_descriptor_distance:
            continue
        if reverse_index is not None and int(reverse_index[int(target_index)]) != source_index:
            continue
        pairs.append((source_index, int(target_index)))
        pair_distance.append(float(distance))
    return _best_pairs(pairs, pair_distance, config.max_correspondences)


def _gated_candidates(source: FeatureSet, target: FeatureSet, config: FeatureConfig, predicted: np.ndarray) -> np.ndarray:
    moved = source.xyz @ predicted[:3, :3].T + predicted[:3, 3]
    nearby = cKDTree(target.xyz).query_ball_point(moved, r=float(config.gate_radius))
    per_keypoint = max(1, int(config.candidates_per_keypoint))
    pairs: list[tuple[int, int]] = []
    pair_distance: list[float] = []
    for source_index, neighbours in enumerate(nearby):
        if not neighbours:
            continue
        neighbours = np.asarray(neighbours, dtype=np.int64)
        distances = np.linalg.norm(target.descriptors[neighbours] - source.descriptors[source_index], axis=1)
        keep = distances <= config.max_descriptor_distance
        neighbours, distances = neighbours[keep], distances[keep]
        for row in np.argsort(distances)[:per_keypoint]:
            pairs.append((source_index, int(neighbours[row])))
            pair_distance.append(float(distances[row]))
    return _best_pairs(pairs, pair_distance, config.max_correspondences)


def _best_pairs(pairs: list[tuple[int, int]], pair_distance: list[float], limit: int) -> np.ndarray:
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    order = np.argsort(np.asarray(pair_distance), kind="stable")[: int(limit)]
    return np.asarray(pairs, dtype=np.int32)[order]
