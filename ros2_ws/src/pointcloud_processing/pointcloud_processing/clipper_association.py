"""Thin wrapper around the official ``clipperpy`` bindings.

The affinity matrix is built by CLIPPER's Euclidean-distance invariant.
This module does not implement a different matcher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClipperConfig:
    """Geometric-consistency parameters passed to CLIPPER's Euclidean invariant.

    ``sigma`` and ``epsilon`` are meters. They need tuning for a real LiDAR.
    """

    sigma: float = 0.05
    epsilon: float = 0.2
    mindist: float = 0.1
    min_inliers: int = 8


@dataclass
class AssociationResult:
    """Inlier pairs selected by CLIPPER.

    ``inliers`` is ``(K, 2)`` with source and target keypoint indices.
    ``scores`` aligns with those rows and comes from CLIPPER's solution vector.
    """

    inliers: np.ndarray
    scores: np.ndarray
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.inliers.shape[0])


def _empty(reason: str, candidates: int) -> AssociationResult:
    return AssociationResult(
        inliers=np.zeros((0, 2), dtype=np.int32),
        scores=np.zeros((0,), dtype=np.float64),
        diagnostics={"candidates": candidates, "inliers": 0, "reason": reason},
    )


def associate(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    candidates: np.ndarray,
    config: ClipperConfig,
) -> AssociationResult:
    """Run official CLIPPER on putative correspondences.

    ``source_xyz`` and ``target_xyz`` are ``(N, 3)`` and ``(M, 3)``.
    ``candidates`` is ``(K, 2)`` of ``(source_index, target_index)``.
    Points are passed to CLIPPER as columns, which is the layout its
    invariant uses (``D.col(i)``).
    """
    source = np.asarray(source_xyz, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(-1, 3)
    pairs = np.asarray(candidates, dtype=np.int32).reshape(-1, 2)
    if pairs.shape[0] < 3:
        return _empty("too_few_candidates", int(pairs.shape[0]))
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= source.shape[0]):
        raise ValueError("source correspondence index is out of range")
    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= target.shape[0]):
        raise ValueError("target correspondence index is out of range")

    import clipperpy

    invariant_params = clipperpy.invariants.EuclideanDistanceParams()
    invariant_params.sigma = float(config.sigma)
    invariant_params.epsilon = float(config.epsilon)
    invariant_params.mindist = float(config.mindist)
    invariant = clipperpy.invariants.EuclideanDistance(invariant_params)
    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.NONZERO
    solver = clipperpy.CLIPPER(invariant, params)
    solver.score_pairwise_consistency(
        np.asfortranarray(source.T),
        np.asfortranarray(target.T),
        np.asfortranarray(pairs),
    )
    solver.solve()
    selected = np.asarray(solver.get_selected_associations())
    if selected.size == 0:
        return _empty("no_consistent_set", int(pairs.shape[0]))
    selected = selected.astype(np.int32).reshape(-1, 2)
    solution = solver.get_solution()
    scores = _inlier_scores(solution, selected.shape[0])
    reason = "ok" if selected.shape[0] >= config.min_inliers else "below_min_inliers"
    if reason != "ok":
        logger.info(
            "CLIPPER kept %d correspondences, below the minimum of %d",
            selected.shape[0],
            config.min_inliers,
        )
    return AssociationResult(
        inliers=selected,
        scores=scores,
        diagnostics={
            "candidates": int(pairs.shape[0]),
            "inliers": int(selected.shape[0]),
            "score": float(getattr(solution, "score", 0.0) or 0.0),
            "reason": reason,
        },
    )


def _inlier_scores(solution: object, count: int) -> np.ndarray:
    nodes = list(getattr(solution, "nodes", []))
    values = np.asarray(getattr(solution, "u", np.zeros(0)), dtype=np.float64).reshape(-1)
    if len(nodes) == count and values.size > 0:
        index = np.asarray(nodes, dtype=np.int64)
        valid = (index >= 0) & (index < values.size)
        scores = np.zeros(count, dtype=np.float64)
        scores[valid] = values[index[valid]]
        return scores
    return np.ones(count, dtype=np.float64)
