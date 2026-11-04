"""FPFH keypoints. Candidate pairs are putative; CLIPPER is tested separately."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pointcloud_processing.features import FeatureConfig, FeatureSet, candidate_correspondences, extract_features


def test_features_and_candidates_exist_for_a_transformed_copy() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(50, 3)) * np.array([1.5, 0.8, 0.6])
    rotation = Rotation.from_rotvec([0.15, -0.05, 0.2]).as_matrix()
    target = (rotation @ source.T).T + np.array([0.3, -0.2, 0.1])
    config = FeatureConfig(
        keypoint_voxel=0.2,
        normal_radius=1.0,
        fpfh_radius=1.6,
        max_correspondences=60,
        max_descriptor_distance=10.0,
        mutual=False,
    )
    source_features = extract_features(source, config)
    target_features = extract_features(target, config)
    assert source_features.count >= 8
    assert source_features.descriptors.shape == (source_features.count, 33)
    assert np.isfinite(source_features.descriptors).all()
    pairs = candidate_correspondences(source_features, target_features, config)
    assert pairs.shape[1] == 2
    assert pairs.shape[0] >= 1


def test_random_keypoints_are_seeded_rows_of_the_full_cloud() -> None:
    rng = np.random.default_rng(8)
    cloud = rng.normal(size=(600, 3)) * np.array([2.0, 2.0, 0.3])
    config = FeatureConfig(keypoint_method="random", max_keypoints=120, normal_radius=0.6, fpfh_radius=1.0)

    first = extract_features(cloud, config)
    second = extract_features(cloud, config)

    assert first.count == 120
    assert first.descriptors.shape == (120, 33)
    assert np.array_equal(first.xyz, second.xyz)
    rows = {tuple(point) for point in cloud}
    assert all(tuple(point) in rows for point in first.xyz)


def test_unknown_keypoint_method_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        extract_features(np.zeros((20, 3)), FeatureConfig(keypoint_method="iss"))


def test_gated_candidates_stay_near_the_predicted_position() -> None:
    rng = np.random.default_rng(6)
    source = FeatureSet(xyz=rng.uniform(-3.0, 3.0, size=(30, 3)), descriptors=rng.uniform(size=(30, 33)))
    motion = np.eye(4)
    motion[:3, 3] = [0.4, 0.0, 0.0]
    target_xyz = source.xyz + motion[:3, 3]
    target = FeatureSet(xyz=target_xyz, descriptors=source.descriptors + rng.normal(0.0, 0.01, size=(30, 33)))
    config = FeatureConfig(max_correspondences=200, max_descriptor_distance=10.0, candidates_per_keypoint=3, gate_radius=0.5)

    pairs = candidate_correspondences(source, target, config, predicted=motion)

    assert pairs.shape[0] >= 30
    moved = source.xyz[pairs[:, 0]] + motion[:3, 3]
    assert np.all(np.linalg.norm(moved - target.xyz[pairs[:, 1]], axis=1) <= 0.5)
    assert {(index, index) for index in range(30)} <= {(int(a), int(b)) for a, b in pairs}
    assert np.max(np.bincount(pairs[:, 0])) <= 3
