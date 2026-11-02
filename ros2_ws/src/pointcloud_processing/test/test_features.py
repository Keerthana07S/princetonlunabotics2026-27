"""FPFH keypoints. Candidate pairs are putative; CLIPPER is tested separately."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pointcloud_processing.features import FeatureConfig, candidate_correspondences, extract_features


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
