"""Rigid registration from CLIPPER inliers, with an optional Open3D ICP refine."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from lidar_interface.geometry import transform_points

logger = logging.getLogger(__name__)


@dataclass
class RegistrationConfig:
    """Acceptance thresholds. A failed registration must not be integrated.

    Without full clouds the gate is on the CLIPPER pairs: RMSE in meters and
    fitness, the fraction of pairs under ``inlier_threshold``. With full
    clouds the gate also requires the aligned clouds to agree: at least
    ``min_overlap`` of source points within ``overlap_distance`` of the
    target, and the RMSE of those points under ``max_rmse``. ``min_fitness``
    then checks that the final transform still agrees with CLIPPER's pairs.
    """

    max_rmse: float = 0.2
    min_fitness: float = 0.4
    inlier_threshold: float = 0.25
    min_points: int = 8
    icp_enabled: bool = True
    icp_distance: float = 0.35
    icp_iterations: int = 30
    icp_point_to_plane: bool = True
    icp_normal_radius: float = 0.6
    overlap_distance: float = 0.3
    min_overlap: float = 0.5


@dataclass
class RegistrationResult:
    """``T_target_source`` maps source points into the target frame: ``p_t = T p_s``.

    ``rmse`` is over the full clouds when they were given, otherwise over the
    CLIPPER pairs. ``overlap`` is 0 when no full clouds were given.
    """

    accepted: bool
    transform: np.ndarray
    rmse: float
    fitness: float
    inlier_count: int
    reason: str
    overlap: float = 0.0


def _rejected(reason: str, count: int = 0) -> RegistrationResult:
    return RegistrationResult(
        accepted=False,
        transform=np.eye(4),
        rmse=float("inf"),
        fitness=0.0,
        inlier_count=count,
        reason=reason,
    )


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Kabsch/SVD rigid transform from matched ``(N, 3)`` points.

    The rotation is forced to a proper rotation (``det(R) = +1``).
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Kabsch needs at least three matched points")
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)
    covariance = src_centered.T @ dst_centered
    _u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ _u.T
    if np.linalg.det(rotation) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rotation = vt.T @ _u.T
    if np.linalg.det(rotation) < 0.0:
        raise RuntimeError("Kabsch produced a reflection")
    translation = dst.mean(axis=0) - rotation @ src.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def alignment_error(source: np.ndarray, target: np.ndarray, transform: np.ndarray, threshold: float) -> tuple[float, float]:
    """Return RMSE and the fraction of pairs closer than ``threshold``."""
    predicted = transform_points(source, transform)
    error = np.linalg.norm(predicted - target, axis=1)
    if error.size == 0:
        return float("inf"), 0.0
    rmse = float(np.sqrt(np.mean(error ** 2)))
    fitness = float(np.mean(error <= threshold))
    return rmse, fitness


def cloud_agreement(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    distance: float,
    target_normals: np.ndarray | None = None,
) -> tuple[float, float]:
    """Residual RMSE of source points within ``distance`` of the target, and the fraction that are.

    With ``target_normals`` the residual is point-to-plane, which stays near
    the sensor noise for correctly aligned surfaces even when the two clouds
    were sampled at different places. Without normals it is point-to-point.
    """
    if source.shape[0] == 0 or target.shape[0] == 0:
        return float("inf"), 0.0
    moved = transform_points(source, transform)
    nearest, index = cKDTree(target).query(moved, k=1, distance_upper_bound=distance)
    close = np.isfinite(nearest)
    if not np.any(close):
        return float("inf"), 0.0
    if target_normals is None:
        residual = nearest[close]
    else:
        matched = index[close]
        residual = np.einsum("ij,ij->i", moved[close] - target[matched], target_normals[matched])
    return float(np.sqrt(np.mean(residual ** 2))), float(np.mean(close))


def estimate_normals(points: np.ndarray, radius: float, max_nn: int = 60) -> np.ndarray:
    """PCA normals. The radius must span several scan rings or ground normals tilt toward the sensor."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    return np.asarray(cloud.normals, dtype=np.float64)


def refine_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    config: RegistrationConfig,
    target_normals: np.ndarray | None = None,
) -> np.ndarray:
    """Point-to-plane (or point-to-point) ICP. The initial matrix is returned when a cloud is too small."""
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 10 or dst.shape[0] < 10:
        return np.asarray(initial, dtype=np.float64)
    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(src)
    target_cloud.points = o3d.utility.Vector3dVector(dst)
    if config.icp_point_to_plane:
        if target_normals is None:
            target_normals = estimate_normals(dst, config.icp_normal_radius)
        target_cloud.normals = o3d.utility.Vector3dVector(target_normals)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        float(config.icp_distance),
        np.asarray(initial, dtype=np.float64),
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(config.icp_iterations)),
    )
    return np.asarray(result.transformation, dtype=np.float64)


def register_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    inliers: np.ndarray,
    config: RegistrationConfig,
    source_cloud: np.ndarray | None = None,
    target_cloud: np.ndarray | None = None,
) -> RegistrationResult:
    """Estimate, optionally refine, and accept or reject a rigid alignment.

    ``inliers`` indexes ``source_points`` and ``target_points``. When cloud
    arguments are provided they are the full scans used only for ICP.
    """
    pairs = np.asarray(inliers, dtype=np.int32).reshape(-1, 2)
    if pairs.shape[0] < config.min_points:
        return _rejected("too_few_inliers", int(pairs.shape[0]))
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    try:
        transform = estimate_rigid_transform(source[pairs[:, 0]], target[pairs[:, 1]])
    except ValueError as exc:
        logger.error("Rigid estimation failed: %s", exc)
        return _rejected("estimation_failed", int(pairs.shape[0]))
    rmse, fitness = alignment_error(
        source[pairs[:, 0]],
        target[pairs[:, 1]],
        transform,
        config.inlier_threshold,
    )
    if source_cloud is None or target_cloud is None:
        accepted = rmse <= config.max_rmse and fitness >= config.min_fitness
        if not accepted:
            logger.info("Rejected registration rmse=%.4f fitness=%.3f", rmse, fitness)
        return RegistrationResult(
            accepted=accepted,
            transform=transform,
            rmse=rmse,
            fitness=fitness,
            inlier_count=int(pairs.shape[0]),
            reason="accepted" if accepted else "quality_gate",
        )

    full_source = np.asarray(source_cloud, dtype=np.float64).reshape(-1, 3)
    full_target = np.asarray(target_cloud, dtype=np.float64).reshape(-1, 3)
    normals = None
    if config.icp_point_to_plane and full_target.shape[0] >= 10:
        normals = estimate_normals(full_target, config.icp_normal_radius)
    candidates = [transform]
    if config.icp_enabled:
        candidates.append(refine_icp(full_source, full_target, transform, config, normals))
    best = None
    for candidate in candidates:
        cloud_rmse, overlap = cloud_agreement(full_source, full_target, candidate, config.overlap_distance, normals)
        if best is None or (overlap, -cloud_rmse) > (best[2], -best[1]):
            best = (candidate, cloud_rmse, overlap)
    transform, rmse, overlap = best
    _pair_rmse, fitness = alignment_error(source[pairs[:, 0]], target[pairs[:, 1]], transform, config.inlier_threshold)
    accepted = rmse <= config.max_rmse and overlap >= config.min_overlap and fitness >= config.min_fitness
    reason = "accepted" if accepted else "quality_gate"
    if not accepted:
        logger.info(
            "Rejected registration cloud_rmse=%.4f overlap=%.3f pair_fitness=%.3f", rmse, overlap, fitness
        )
    return RegistrationResult(
        accepted=accepted,
        transform=transform,
        rmse=rmse,
        fitness=fitness,
        inlier_count=int(pairs.shape[0]),
        reason=reason,
        overlap=overlap,
    )
