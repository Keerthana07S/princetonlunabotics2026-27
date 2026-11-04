"""CLIPPER must keep a rigid clique and drop planted outliers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("clipperpy")

from pointcloud_processing.clipper_association import ClipperConfig, associate
from pointcloud_processing.registration import estimate_rigid_transform


def _shape(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(12, 3))
    points *= np.array([2.0, 1.4, 1.1])
    return points


def test_clipper_rejects_planted_outlier_correspondences() -> None:
    source = _shape(1)
    rotation = Rotation.from_rotvec([0.1, 0.2, 0.5]).as_matrix()
    translation = np.array([1.0, -0.4, 0.25])
    target = (rotation @ source.T).T + translation
    true_pairs = np.stack([np.arange(len(source)), np.arange(len(source))], axis=1)
    rng = np.random.default_rng(2)
    outliers = np.stack(
        [rng.integers(0, len(source), size=24), rng.integers(0, len(source), size=24)],
        axis=1,
    )
    outliers = outliers[outliers[:, 0] != outliers[:, 1]]
    candidates = np.vstack([true_pairs, outliers]).astype(np.int32)
    result = associate(
        source,
        target,
        candidates,
        ClipperConfig(sigma=0.01, epsilon=0.05, mindist=0.0, min_inliers=6),
    )
    assert result.count >= 8
    assert result.count < candidates.shape[0]
    selected = {(int(pair[0]), int(pair[1])) for pair in result.inliers}
    true = {(index, index) for index in range(len(source))}
    true_positive = len(selected & true)
    assert true_positive >= 8
    assert true_positive / result.count >= 0.8
    assert result.diagnostics["reason"] == "ok"


def test_clipper_with_noise_and_outliers_recovers_the_transform() -> None:
    rng = np.random.default_rng(21)
    source = rng.uniform(-4.0, 4.0, size=(40, 3)) * np.array([1.0, 1.0, 0.4])
    rotation = Rotation.from_euler("xyz", [0.03, -0.02, 0.35]).as_matrix()
    translation = np.array([0.8, -0.3, 0.05])
    target = (rotation @ source.T).T + translation
    target = target + rng.normal(0.0, 0.02, size=target.shape)
    true_pairs = np.stack([np.arange(len(source)), np.arange(len(source))], axis=1)
    outliers = np.stack([rng.integers(0, len(source), 160), rng.integers(0, len(source), 160)], axis=1)
    outliers = outliers[outliers[:, 0] != outliers[:, 1]]
    candidates = np.vstack([true_pairs, outliers]).astype(np.int32)
    candidates = candidates[rng.permutation(candidates.shape[0])]

    result = associate(source, target, candidates, ClipperConfig(sigma=0.04, epsilon=0.12, mindist=0.0, min_inliers=6))

    assert result.diagnostics["reason"] == "ok"
    selected = {(int(pair[0]), int(pair[1])) for pair in result.inliers}
    true_positive = len(selected & {(index, index) for index in range(len(source))})
    assert true_positive >= 30
    assert true_positive / result.count >= 0.9
    estimate = estimate_rigid_transform(source[result.inliers[:, 0]], target[result.inliers[:, 1]])
    rotation_error = Rotation.from_matrix(estimate[:3, :3] @ rotation.T).magnitude()
    assert np.degrees(rotation_error) < 1.0
    assert np.linalg.norm(estimate[:3, 3] - translation) < 0.05


def test_clipper_finds_no_large_set_in_an_unrelated_cloud() -> None:
    source = _shape(3)
    target = _shape(9)
    candidates = np.stack([np.arange(len(source)), np.arange(len(target))], axis=1)
    result = associate(
        source,
        target,
        candidates,
        ClipperConfig(sigma=0.01, epsilon=0.05, mindist=0.0, min_inliers=6),
    )
    assert result.count <= 4
    assert result.diagnostics["reason"] != "ok"
