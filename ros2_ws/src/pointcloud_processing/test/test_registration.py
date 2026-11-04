"""Kabsch, reflection rejection, and the quality gate."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pointcloud_processing.registration import (
    RegistrationConfig,
    alignment_error,
    estimate_rigid_transform,
    register_correspondences,
)


def _cloud() -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.normal(size=(30, 3))


def test_kabsch_recovers_a_known_rigid_motion() -> None:
    source = _cloud()
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    translation = np.array([0.8, -0.3, 0.15])
    target = (rotation @ source.T).T + translation
    pairs = np.stack([np.arange(len(source)), np.arange(len(source))], axis=1)
    transform = estimate_rigid_transform(source, target)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(transform[:3, :3], rotation, atol=1e-6)
    assert np.allclose(transform[:3, 3], translation, atol=1e-6)
    rmse, fitness = alignment_error(source, target, transform, 0.05)
    assert rmse < 1e-6
    assert fitness == 1.0
    result = register_correspondences(
        source,
        target,
        pairs,
        RegistrationConfig(icp_enabled=False, max_rmse=0.05, min_fitness=0.9, min_points=8),
    )
    assert result.accepted
    assert result.reason == "accepted"


def test_kabsch_returns_a_proper_rotation_for_a_mirrored_set() -> None:
    source = _cloud()
    mirrored = source.copy()
    mirrored[:, 0] *= -1.0
    transform = estimate_rigid_transform(source, mirrored)
    assert np.linalg.det(transform[:3, :3]) > 0.0


def test_quality_gate_rejects_a_bad_alignment_without_claiming_success() -> None:
    source = _cloud()
    target = np.random.default_rng(99).normal(size=source.shape)
    pairs = np.stack([np.arange(10), np.arange(10)], axis=1)
    result = register_correspondences(
        source,
        target,
        pairs,
        RegistrationConfig(icp_enabled=False, max_rmse=0.05, min_fitness=0.95, min_points=8),
    )
    assert not result.accepted
    assert result.reason == "quality_gate"
