"""Rigid registration from CLIPPER inliers, with an optional Open3D ICP refine."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from lidar_interface.geometry import transform_points

logger = logging.getLogger(__name__)


@dataclass
class RegistrationConfig:
    """Acceptance thresholds. A failed registration must not be integrated.

    RMSE is meters. Fitness is the fraction of inlier pairs under
    ``inlier_threshold``.
    """

    max_rmse: float = 0.2
    min_fitness: float = 0.4
    inlier_threshold: float = 0.25
    min_points: int = 8
    icp_enabled: bool = True
    icp_distance: float = 0.35
    icp_iterations: int = 30


@dataclass
class RegistrationResult:
    """``T_target_source`` maps source points into the target frame: ``p_t = T p_s``."""

    accepted: bool
    transform: np.ndarray
    rmse: float
    fitness: float
    inlier_count: int
    reason: str


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


def refine_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    """Point-to-point ICP. The initial matrix is returned when a cloud is too small."""
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 10 or dst.shape[0] < 10:
        return np.asarray(initial, dtype=np.float64)
    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(src)
    target_cloud.points = o3d.utility.Vector3dVector(dst)
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        float(config.icp_distance),
        np.asarray(initial, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
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
    if config.icp_enabled and source_cloud is not None and target_cloud is not None:
        refined = refine_icp(source_cloud, target_cloud, transform, config)
        refined_rmse, refined_fitness = alignment_error(
            source[pairs[:, 0]],
            target[pairs[:, 1]],
            refined,
            config.inlier_threshold,
        )
        if refined_rmse <= rmse:
            transform = refined
            rmse = refined_rmse
            fitness = refined_fitness
    accepted = rmse <= config.max_rmse and fitness >= config.min_fitness
    reason = "accepted" if accepted else "quality_gate"
    if not accepted:
        logger.info("Rejected registration rmse=%.4f fitness=%.3f", rmse, fitness)
    return RegistrationResult(
        accepted=accepted,
        transform=transform,
        rmse=rmse,
        fitness=fitness,
        inlier_count=int(pairs.shape[0]),
        reason=reason,
    )
