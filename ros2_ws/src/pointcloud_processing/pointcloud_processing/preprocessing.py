"""Open3D preprocessing. Empty clouds skip the expensive steps."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from lidar_interface.geometry import transform_points

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Filters applied to one scan. Distances are meters.

    ``voxel_size``, the outlier radii, and the range limits need tuning for a
    real sensor. The defaults target the synthetic scene on a CPU.
    """

    min_range: float = 0.2
    max_range: float = 20.0
    voxel_size: float = 0.15
    statistical_enabled: bool = True
    statistical_neighbors: int = 16
    statistical_std_ratio: float = 2.0
    radius_enabled: bool = False
    radius: float = 0.4
    radius_min_neighbors: int = 4
    segment_enabled: bool = False
    cluster_eps: float = 0.35
    cluster_min_points: int = 8
    min_cluster_size: int = 30
    plane_removal_enabled: bool = False
    plane_distance: float = 0.08


def _as_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def _as_array(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    if len(cloud.points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(cloud.points, dtype=np.float64)


def filter_range(points: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Keep points whose distance from the sensor origin is inside the range."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return xyz
    distance = np.linalg.norm(xyz, axis=1)
    return xyz[(distance >= minimum) & (distance <= maximum)]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return one representative point per voxel. Non-positive size is a no-op."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0 or voxel_size <= 0.0:
        return xyz
    return _as_array(_as_cloud(xyz).voxel_down_sample(voxel_size))


def statistical_outlier_removal(points: np.ndarray, neighbors: int, std_ratio: float) -> np.ndarray:
    """Remove points whose neighbor distance is an outlier. Small clouds are kept."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] <= neighbors or neighbors < 2:
        return xyz
    cleaned, _index = _as_cloud(xyz).remove_statistical_outlier(
        nb_neighbors=int(neighbors),
        std_ratio=float(std_ratio),
    )
    return _as_array(cleaned)


def radius_outlier_removal(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """Remove points with fewer than ``min_neighbors`` inside ``radius``."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0 or radius <= 0.0:
        return xyz
    cleaned, _index = _as_cloud(xyz).remove_radius_outlier(
        nb_points=int(min_neighbors),
        radius=float(radius),
    )
    return _as_array(cleaned)


def segment_clusters(points: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    """Euclidean clusters. Noise points labeled -1 by DBSCAN are omitted."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return []
    labels = np.asarray(
        _as_cloud(xyz).cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False)
    )
    clusters = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        clusters.append(xyz[labels == label])
    return clusters


def remove_dominant_plane(points: np.ndarray, distance: float) -> np.ndarray:
    """Drop the largest RANSAC plane. Intended to be off for lunar terrain."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] < 50:
        return xyz
    cloud = _as_cloud(xyz)
    _model, inliers = cloud.segment_plane(
        distance_threshold=float(distance),
        ransac_n=3,
        num_iterations=80,
    )
    if len(inliers) < 50:
        return xyz
    return _as_array(cloud.select_by_index(inliers, invert=True))


def preprocess(points: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Run the configured filters. The input is not modified."""
    xyz = filter_range(points, config.min_range, config.max_range)
    if xyz.shape[0] == 0:
        logger.info("Preprocessing received no points inside the range gate")
        return xyz
    xyz = voxel_downsample(xyz, config.voxel_size)
    if config.statistical_enabled:
        xyz = statistical_outlier_removal(xyz, config.statistical_neighbors, config.statistical_std_ratio)
    if config.radius_enabled:
        xyz = radius_outlier_removal(xyz, config.radius, config.radius_min_neighbors)
    if config.plane_removal_enabled:
        xyz = remove_dominant_plane(xyz, config.plane_distance)
    if config.segment_enabled and xyz.shape[0] > 0:
        clusters = segment_clusters(xyz, config.cluster_eps, config.cluster_min_points)
        kept = [cluster for cluster in clusters if cluster.shape[0] >= config.min_cluster_size]
        xyz = np.vstack(kept) if kept else np.zeros((0, 3))
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def transform_cloud(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply ``p_target = T_target_source p_source``."""
    return transform_points(points, transform)
