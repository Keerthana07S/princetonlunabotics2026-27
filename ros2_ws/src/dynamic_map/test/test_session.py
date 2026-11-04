"""Session gates and a known rigid motion through CLIPPER."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("clipperpy")

from dynamic_map.session import MappingSession, SessionConfig
from dynamic_map.voxel_map import MapConfig
from lidar_interface.geometry import invert_transform, make_transform, yaw_rotation
from pointcloud_processing.clipper_association import ClipperConfig
from pointcloud_processing.features import FeatureConfig
from pointcloud_processing.registration import RegistrationConfig


def _config(mode: str = "scan_to_scan") -> SessionConfig:
    return SessionConfig(
        features=FeatureConfig(
            keypoint_voxel=0.15,
            normal_radius=1.2,
            fpfh_radius=2.0,
            max_correspondences=120,
            max_descriptor_distance=8.0,
            mutual=False,
        ),
        clipper=ClipperConfig(sigma=0.02, epsilon=0.15, mindist=0.0, min_inliers=6),
        registration=RegistrationConfig(
            max_rmse=0.25,
            min_fitness=0.3,
            inlier_threshold=0.3,
            min_points=6,
            icp_enabled=True,
            icp_distance=0.5,
            icp_iterations=40,
        ),
        voxel=MapConfig(resolution=0.08, min_static_observations=1, observation_weight=1.0, decay_lambda=0.0),
        mode=mode,
        min_points=20,
    )


def _cloud() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(size=(90, 3)) * np.array([2.0, 1.2, 0.7])


def test_rejected_scan_does_not_change_the_map() -> None:
    session = MappingSession(_config())
    first = session.process_scan(_cloud(), 0.0)
    assert first.accepted
    before = session.map.snapshot()
    rejected = session.process_scan(np.zeros((3, 3)), 1.0)
    assert not rejected.accepted
    assert rejected.reason == "too_few_points"
    assert session.map.snapshot() == before
    assert np.allclose(session.transform, np.eye(4))


def test_coasting_rejection_advances_the_pose_but_not_the_map() -> None:
    from dataclasses import replace

    from lidar_interface.geometry import transform_points

    session = MappingSession(replace(_config(), coast_on_reject=True))
    cloud = _cloud()
    motion = make_transform(yaw_rotation(0.0), np.array([0.3, 0.0, 0.0]))
    assert session.process_scan(cloud, 0.0).accepted
    assert session.process_scan(transform_points(cloud, invert_transform(motion)), 1.0).accepted
    before = session.map.snapshot()
    pose = session.transform.copy()

    unrelated = np.random.default_rng(99).uniform(-4.0, 4.0, size=(90, 3))
    rejected = session.process_scan(unrelated, 2.0)

    assert not rejected.accepted
    assert session.map.snapshot() == before
    step = np.linalg.inv(pose) @ session.transform
    assert np.allclose(step[:3, 3], motion[:3, 3], atol=0.15)
    assert np.allclose(rejected.transform, session.transform)


def test_without_coasting_a_rejection_keeps_the_pose() -> None:
    session = MappingSession(_config())
    assert session.process_scan(_cloud(), 0.0).accepted
    unrelated = np.random.default_rng(99).uniform(-4.0, 4.0, size=(90, 3))
    assert not session.process_scan(unrelated, 1.0).accepted
    assert np.allclose(session.transform, np.eye(4))


def test_first_scan_is_placed_at_the_sensor_extrinsic() -> None:
    from dynamic_map.params import session_config

    session = MappingSession(session_config({"lidar_x": 0.28, "lidar_y": 0.08, "lidar_z": 0.13}))
    rng = np.random.default_rng(3)
    cloud = rng.uniform(-3.0, 3.0, size=(400, 3))
    result = session.process_scan(cloud, 0.0)
    assert result.accepted
    assert np.allclose(result.transform[:3, 3], [0.28, 0.08, 0.13])


def test_scan_to_scan_recovers_a_known_motion() -> None:
    session = MappingSession(_config("scan_to_scan"))
    cloud = _cloud()
    assert session.process_scan(cloud, 0.0).accepted
    motion = make_transform(yaw_rotation(0.12), np.array([0.35, 0.05, 0.0]))
    moved = invert_transform(motion)  # points expressed in the new sensor frame
    from lidar_interface.geometry import transform_points

    second = session.process_scan(transform_points(cloud, moved), 1.0)
    assert second.accepted, second.reason
    assert np.allclose(second.transform[:3, 3], motion[:3, 3], atol=0.15)
    assert abs(np.linalg.det(second.transform[:3, :3]) - 1.0) < 1e-6


def test_scan_to_submap_recovers_a_known_motion() -> None:
    session = MappingSession(_config("scan_to_submap"))
    cloud = _cloud()
    assert session.process_scan(cloud, 0.0).accepted
    motion = make_transform(yaw_rotation(-0.08), np.array([0.25, -0.04, 0.0]))
    from lidar_interface.geometry import transform_points

    second = session.process_scan(transform_points(cloud, invert_transform(motion)), 1.0)
    assert second.accepted, second.reason
    assert np.allclose(second.transform[:3, 3], motion[:3, 3], atol=0.15)


def test_default_settings_track_the_synthetic_terrain_drive() -> None:
    from dynamic_map.params import session_config
    from lidar_interface.synthetic_scene import LidarModel, demo_scene, line_pose
    from pointcloud_processing.preprocessing import PreprocessConfig, preprocess

    scene = demo_scene()
    model = LidarModel(vertical_min_deg=-25.0, vertical_max_deg=15.0, azimuth_bins=240, elevation_bins=16, seed=3)
    session = MappingSession(session_config({}))
    start = None
    accepted = 0
    for index in range(11):
        stamp = 0.5 * index
        pose = scene.sensor_pose(line_pose(stamp, 0.3))
        start = pose.transform if start is None else start
        cloud = preprocess(scene.scan(pose, stamp, model), PreprocessConfig(voxel_size=0.2, statistical_neighbors=12))
        accepted += session.process_scan(cloud, stamp).accepted
    truth = np.linalg.inv(start) @ pose.transform
    assert accepted >= 9
    assert np.linalg.norm(session.transform[:3, 3] - truth[:3, 3]) < 0.15
