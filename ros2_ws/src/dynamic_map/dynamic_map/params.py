"""Mapping parameters shared by the ROS node and the offline benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from dynamic_map.session import SessionConfig
from dynamic_map.voxel_map import MapConfig
from pointcloud_processing.clipper_association import ClipperConfig
from pointcloud_processing.features import FeatureConfig
from pointcloud_processing.registration import RegistrationConfig

DEFAULTS: dict[str, Any] = {
    "input_topic": "/mapping/processed_scan",
    "registered_topic": "/mapping/registered_scan",
    "map_topic": "/mapping/global_map",
    "pose_topic": "/mapping/rover_pose",
    "status_topic": "/mapping/registration_status",
    "correspondence_topic": "/mapping/correspondences",
    "map_frame": "map",
    "base_frame": "base_link",
    "map_reliable": True,
    "registration_mode": "scan_to_scan",
    "submap_radius": 8.0,
    "min_points": 30,
    "motion_prediction": True,
    "lidar_x": 0.0,
    "lidar_y": 0.0,
    "lidar_z": 0.0,
    "lidar_yaw": 0.0,
    "keypoint_voxel": 0.35,
    "normal_radius": 0.8,
    "fpfh_radius": 1.2,
    "max_correspondences": 400,
    "max_descriptor_distance": 2.0,
    "mutual_correspondences": False,
    "candidates_per_keypoint": 3,
    "gate_radius": 1.0,
    "clipper_sigma": 0.05,
    "clipper_epsilon": 0.25,
    "clipper_mindist": 0.05,
    "clipper_min_inliers": 6,
    "max_rmse": 0.06,
    "min_fitness": 0.2,
    "inlier_threshold": 0.4,
    "icp_enabled": True,
    "icp_distance": 0.5,
    "icp_iterations": 50,
    "icp_point_to_plane": True,
    "icp_normal_radius": 1.5,
    "overlap_distance": 0.3,
    "min_overlap": 0.5,
    "voxel_resolution": 0.2,
    "observation_weight": 0.35,
    "decay_lambda": 0.05,
    "max_confidence": 2.0,
    "min_confidence": 0.15,
    "max_voxels": 200000,
    "min_static_observations": 4,
    "motion_threshold": 0.15,
    "free_weight": 0.3,
    "free_hits_to_clear": 2,
    "free_margin": 0.3,
    "free_azimuth_deg": 2.0,
    "free_elevation_deg": 3.0,
    "metrics_path": "/workspace/logs/metrics.jsonl",
}


def session_config(values: Mapping[str, Any]) -> SessionConfig:
    """Build a :class:`SessionConfig` from flat parameter values; missing keys use ``DEFAULTS``."""
    merged = {**DEFAULTS, **values}

    def number(name: str) -> float:
        return float(merged[name])

    def integer(name: str) -> int:
        return int(merged[name])

    return SessionConfig(
        features=FeatureConfig(
            keypoint_voxel=number("keypoint_voxel"),
            normal_radius=number("normal_radius"),
            fpfh_radius=number("fpfh_radius"),
            max_correspondences=integer("max_correspondences"),
            max_descriptor_distance=number("max_descriptor_distance"),
            mutual=bool(merged["mutual_correspondences"]),
            candidates_per_keypoint=integer("candidates_per_keypoint"),
            gate_radius=number("gate_radius"),
        ),
        clipper=ClipperConfig(
            sigma=number("clipper_sigma"),
            epsilon=number("clipper_epsilon"),
            mindist=number("clipper_mindist"),
            min_inliers=integer("clipper_min_inliers"),
        ),
        registration=RegistrationConfig(
            max_rmse=number("max_rmse"),
            min_fitness=number("min_fitness"),
            inlier_threshold=number("inlier_threshold"),
            min_points=integer("clipper_min_inliers"),
            icp_enabled=bool(merged["icp_enabled"]),
            icp_distance=number("icp_distance"),
            icp_iterations=integer("icp_iterations"),
            icp_point_to_plane=bool(merged["icp_point_to_plane"]),
            icp_normal_radius=number("icp_normal_radius"),
            overlap_distance=number("overlap_distance"),
            min_overlap=number("min_overlap"),
        ),
        voxel=MapConfig(
            resolution=number("voxel_resolution"),
            observation_weight=number("observation_weight"),
            decay_lambda=number("decay_lambda"),
            max_confidence=number("max_confidence"),
            min_confidence=number("min_confidence"),
            max_voxels=integer("max_voxels"),
            min_static_observations=integer("min_static_observations"),
            motion_threshold=number("motion_threshold"),
            free_weight=number("free_weight"),
            free_hits_to_clear=integer("free_hits_to_clear"),
            free_margin=number("free_margin"),
            free_azimuth_deg=number("free_azimuth_deg"),
            free_elevation_deg=number("free_elevation_deg"),
        ),
        mode=str(merged["registration_mode"]),
        submap_radius=number("submap_radius"),
        min_points=integer("min_points"),
        motion_prediction=bool(merged["motion_prediction"]),
    )


def load_node_parameters(paths: list[Path], node: str) -> dict[str, Any]:
    """Merge ``<node>.ros__parameters`` from several ROS parameter files, later files winning."""
    values: dict[str, Any] = {}
    for path in paths:
        content = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values.update((content.get(node) or {}).get("ros__parameters") or {})
    return values
