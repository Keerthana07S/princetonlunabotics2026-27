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


def candidate_correspondences(source: FeatureSet, target: FeatureSet, config: FeatureConfig) -> np.ndarray:
    """Mutual nearest neighbors in FPFH space, capped at ``max_correspondences``.

    The result is an ``(K, 2)`` int array of ``(source_index, target_index)``.
    This is a putative set. It is not the CLIPPER inlier decision.
    """
    if source.count == 0 or target.count == 0:
        return np.zeros((0, 2), dtype=np.int32)
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
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    order = np.argsort(np.asarray(pair_distance))[: int(config.max_correspondences)]
    return np.asarray(pairs, dtype=np.int32)[order]
