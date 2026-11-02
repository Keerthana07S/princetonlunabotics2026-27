#!/usr/bin/env python3
"""Run the library pipeline over the whole synthetic trajectory and score it.

Settings come from the same config/*.yaml files the launch uses. Pose error is
measured against the simulator's ground truth, expressed in the map frame
(the sensor pose of the first scan). Every number written here is measured by
this process; none are estimates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dynamic_map.params import load_node_parameters, session_config
from dynamic_map.session import MappingSession
from lidar_interface.synthetic_scene import DEMO_PATH_LENGTH, Box, LidarModel, Sphere, demo_scene, line_pose
from pointcloud_processing.preprocessing import PreprocessConfig, preprocess

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return -1.0
    return -1.0


def lidar_model() -> LidarModel:
    values = load_node_parameters([CONFIG / "lidar.yaml"], "synthetic_lidar")
    return LidarModel(
        min_range=float(values["min_range"]),
        max_range=float(values["max_range"]),
        horizontal_fov_deg=float(values["horizontal_fov_deg"]),
        horizontal_center_deg=float(values["horizontal_center_deg"]),
        vertical_min_deg=float(values["vertical_min_deg"]),
        vertical_max_deg=float(values["vertical_max_deg"]),
        azimuth_bins=int(values["azimuth_bins"]),
        elevation_bins=int(values["elevation_bins"]),
        range_noise_std=float(values["range_noise_std"]),
        seed=int(values["seed"]),
    )


def preprocess_config() -> PreprocessConfig:
    values = load_node_parameters([CONFIG / "preprocessing.yaml"], "preprocessing")
    fields = PreprocessConfig.__dataclass_fields__
    return PreprocessConfig(**{key: value for key, value in values.items() if key in fields})


def mapping_values(overrides: list[str]) -> dict:
    files = [CONFIG / name for name in ("mapping.yaml", "features.yaml", "clipper.yaml", "registration.yaml")]
    values = load_node_parameters(files, "mapping")
    for item in overrides:
        key, raw = item.split("=", 1)
        values[key] = json.loads(raw) if raw[:1] in "0123456789-[{tf" else raw
    return values


def object_voxels(world_points: np.ndarray, primitive: Sphere | Box, terrain, margin: float = 0.15) -> int:
    """Map voxels on ``primitive`` (world frame) and clearly above the terrain under it."""
    if world_points.shape[0] == 0:
        return 0
    if isinstance(primitive, Sphere):
        near = np.linalg.norm(world_points - primitive.center, axis=1) <= primitive.radius + margin
    else:
        near = np.all(
            (world_points >= primitive.minimum - margin) & (world_points <= primitive.maximum + margin), axis=1
        )
    above = world_points[:, 2] - terrain.height(world_points[:, 0], world_points[:, 1]) > 0.15
    return int(np.sum(near & above))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=int, default=0, help="0 drives the full path plus 4 s parked")
    parser.add_argument("--rate", type=float, default=2.0, help="scan rate in Hz")
    parser.add_argument("--speed", type=float, default=0.3)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="mapping parameter key=value")
    parser.add_argument("--output", default="/workspace/logs/benchmark_trajectory")
    args = parser.parse_args()

    dt = 1.0 / args.rate
    scans = args.scans or int(np.ceil((DEMO_PATH_LENGTH / args.speed + 4.0) / dt)) + 1
    scene = demo_scene()
    model = lidar_model()
    pre = preprocess_config()
    values = mapping_values(args.overrides)
    session = MappingSession(session_config(values))

    objects = {primitive.name: primitive for primitive in scene.primitives if isinstance(primitive, (Sphere, Box))}
    mover = objects["mover"]
    mover_start = Sphere("mover_start", mover.center_at(0.0), mover.radius)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    first_inverse = None
    start_pose = np.eye(4)
    for index in range(scans):
        stamp = index * dt
        truth = scene.sensor_pose(line_pose(stamp, args.speed, DEMO_PATH_LENGTH))
        if first_inverse is None:
            start_pose = truth.transform
            first_inverse = np.linalg.inv(start_pose)
        truth_map = first_inverse @ truth.transform
        generated = time.perf_counter()
        raw = scene.scan(truth, stamp, model)
        generate_s = time.perf_counter() - generated
        filtered_at = time.perf_counter()
        filtered = preprocess(raw, pre)
        preprocess_s = time.perf_counter() - filtered_at
        result = session.process_scan(filtered, stamp)
        error = float(np.linalg.norm(result.transform[:3, 3] - truth_map[:3, 3]))
        exported = session.map.export(stamp)
        world = exported.xyz @ start_pose[:3, :3].T + start_pose[:3, 3]
        record = {
            "scan": index,
            "stamp": stamp,
            "input_points": int(raw.shape[0]),
            "processed_points": int(filtered.shape[0]),
            "generate_s": generate_s,
            "preprocess_s": preprocess_s,
            "accepted": result.accepted,
            "reason": result.reason,
            "rmse": result.rmse,
            "overlap": result.overlap,
            "inliers": result.inliers,
            "voxels": len(session.map.voxels),
            "pose": result.transform[:3, 3].round(4).tolist(),
            "truth": truth_map[:3, 3].round(4).tolist(),
            "translation_error": error,
            "arrival_voxels": object_voxels(world, objects["arrival"], scene.terrain),
            "temporary_voxels": object_voxels(world, objects["temporary"], scene.terrain),
            "mover_voxels_at_start": object_voxels(world, mover_start, scene.terrain),
            "rss_mb": rss_mb(),
        }
        record.update({key: float(value) for key, value in result.timings.items()})
        records.append(record)
        print(
            f"scan {index:3d} t={stamp:5.1f} {result.reason:18s} inliers={result.inliers:3d} "
            f"rmse={result.rmse:6.3f} overlap={result.overlap:4.2f} "
            f"x={result.transform[0, 3]:6.3f} truth={truth_map[0, 3]:6.3f} err={error:.3f} "
            f"total={result.timings.get('total_s', 0.0) * 1000:6.1f} ms voxels={len(session.map.voxels)}",
            flush=True,
        )

    cloud = session.map.export((scans - 1) * dt)
    arrival_time = objects["arrival"].active_start
    vanish_time = objects["temporary"].active_end

    def peak(key: str, start: float, end: float) -> int:
        values = [row[key] for row in records if start <= row["stamp"] < end]
        return int(max(values)) if values else 0

    errors = np.array([row["translation_error"] for row in records])
    accepted = [row for row in records if row["accepted"]]
    totals = np.array([row["total_s"] for row in records if "total_s" in row])

    def mean(key: str) -> float | None:
        rows = [float(row[key]) for row in accepted if key in row]
        return float(np.mean(rows)) if rows else None

    summary = {
        "scans": len(records),
        "scan_rate_hz": args.rate,
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "rejections": sorted({row["reason"] for row in records if not row["accepted"]}),
        "final_pose": records[-1]["pose"],
        "final_truth": records[-1]["truth"],
        "final_translation_error_m": float(errors[-1]),
        "max_translation_error_m": float(errors.max()),
        "rmse_translation_error_m": float(np.sqrt(np.mean(errors ** 2))),
        "path_length_m": float(np.linalg.norm(np.array(records[-1]["truth"]))),
        "final_voxels": len(session.map.voxels),
        "final_static_voxels": int(np.sum(~cloud.dynamic)),
        "arrival_voxels_before_it_appears": peak("arrival_voxels", 0.0, arrival_time),
        "arrival_voxels_final": records[-1]["arrival_voxels"],
        "temporary_voxels_peak_while_present": peak("temporary_voxels", 0.0, vanish_time),
        "temporary_voxels_final": records[-1]["temporary_voxels"],
        "mover_voxels_at_start_peak": peak("mover_voxels_at_start", 0.0, float("inf")),
        "mover_voxels_at_start_final": records[-1]["mover_voxels_at_start"],
        "final_rss_mb": records[-1]["rss_mb"],
        "mean_input_points": float(np.mean([row["input_points"] for row in records])),
        "mean_processed_points": float(np.mean([row["processed_points"] for row in records])),
        "mean_generate_s": float(np.mean([row["generate_s"] for row in records])),
        "mean_preprocess_s": float(np.mean([row["preprocess_s"] for row in records])),
        "mean_features_s": mean("features_s"),
        "mean_candidates_s": mean("candidates_s"),
        "mean_clipper_s": mean("clipper_s"),
        "mean_registration_s": mean("registration_s"),
        "mean_map_s": mean("map_s"),
        "mean_session_total_s": float(np.mean(totals)) if totals.size else None,
        "max_session_total_s": float(np.max(totals)) if totals.size else None,
        "parameters": values,
    }
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    output.with_name(output.name + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "parameters"}, indent=2))


if __name__ == "__main__":
    main()
