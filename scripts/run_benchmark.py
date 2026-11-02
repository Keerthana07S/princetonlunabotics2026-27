#!/usr/bin/env python3
"""Run the library pipeline on the synthetic scene and append measured timings.

The numbers written to logs/benchmark.jsonl are from this process. They are
not estimates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dynamic_map.session import MappingSession, SessionConfig
from dynamic_map.voxel_map import MapConfig
from lidar_interface.synthetic_scene import LidarModel, demo_scene, line_pose
from pointcloud_processing.clipper_association import ClipperConfig
from pointcloud_processing.features import FeatureConfig
from pointcloud_processing.preprocessing import PreprocessConfig, preprocess
from pointcloud_processing.registration import RegistrationConfig


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return -1.0
    return -1.0


def main() -> None:
    output = Path("/workspace/logs/benchmark.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene = demo_scene()
    model = LidarModel(
        min_range=0.2,
        max_range=20.0,
        azimuth_bins=90,
        elevation_bins=8,
        range_noise_std=0.01,
        seed=3,
    )
    preprocess_config = PreprocessConfig(
        voxel_size=0.2,
        statistical_neighbors=12,
        statistical_enabled=True,
        radius_enabled=False,
    )
    session = MappingSession(
        SessionConfig(
            features=FeatureConfig(
                keypoint_voxel=0.35,
                normal_radius=0.8,
                fpfh_radius=1.2,
                max_correspondences=100,
                max_descriptor_distance=2.0,
                mutual=False,
            ),
            clipper=ClipperConfig(sigma=0.05, epsilon=0.25, mindist=0.05, min_inliers=6),
            registration=RegistrationConfig(
                max_rmse=0.12,
                min_fitness=0.5,
                inlier_threshold=0.4,
                min_points=6,
                icp_enabled=True,
                icp_distance=0.5,
            ),
            voxel=MapConfig(resolution=0.25, min_static_observations=3, decay_lambda=0.02),
            mode="scan_to_scan",
            min_points=25,
        )
    )
    records = []
    for index in range(8):
        stamp = index * 0.5
        generated = time.perf_counter()
        raw = scene.scan(line_pose(stamp, speed=0.3), stamp, model)
        generate_s = time.perf_counter() - generated
        filtered_at = time.perf_counter()
        filtered = preprocess(raw, preprocess_config)
        preprocess_s = time.perf_counter() - filtered_at
        result = session.process_scan(filtered, stamp)
        record = {
            "scan": index,
            "input_points": int(raw.shape[0]),
            "processed_points": int(filtered.shape[0]),
            "generate_s": generate_s,
            "preprocess_s": preprocess_s,
            "accepted": result.accepted,
            "reason": result.reason,
            "rmse": result.rmse,
            "inliers": result.inliers,
            "voxels": len(session.map.voxels),
            "static_points": int(session.map.static_points(stamp).shape[0]),
            "pose_x": float(result.transform[0, 3]),
            "truth_x": float(line_pose(stamp, speed=0.3).position[0]),
            "rss_mb": rss_mb(),
        }
        record.update({key: float(value) for key, value in result.timings.items()})
        records.append(record)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
    def _mean(key: str) -> float | None:
        values = [float(row[key]) for row in records if key in row]
        if not values:
            return None
        return float(np.mean(values))

    summary = {
        "scans": len(records),
        "accepted": sum(1 for row in records if row["accepted"]),
        "rejected": sum(1 for row in records if not row["accepted"]),
        "final_voxels": records[-1]["voxels"] if records else 0,
        "final_static_points": records[-1]["static_points"] if records else 0,
        "final_rss_mb": records[-1]["rss_mb"] if records else -1,
        "mean_input_points": _mean("input_points"),
        "mean_preprocess_s": _mean("preprocess_s"),
        "mean_features_s": _mean("features_s"),
        "mean_clipper_s": _mean("clipper_s"),
        "mean_registration_s": _mean("registration_s"),
        "mean_map_s": _mean("map_s"),
        "final_pose_x": records[-1]["pose_x"] if records else None,
        "final_truth_x": records[-1]["truth_x"] if records else None,
    }
    summary_path = Path("/workspace/logs/benchmark_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
