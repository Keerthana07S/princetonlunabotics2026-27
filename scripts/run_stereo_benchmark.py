#!/usr/bin/env python3
"""Run the stereo pipeline over a LAC log without ROS and score it.

Each FrontLeft/FrontRight pair goes through the same code and the same
config files as ``stereo_pipeline.launch.py``: SGBM stereo, preprocessing,
random keypoints and FPFH, CLIPPER, Kabsch, ICP, the quality gate, and the
voxel map. Pose error is against the log's rover pose, expressed relative to
the rover pose of the first pair (the map frame). Every number written here is
measured by this process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dynamic_map.params import load_node_parameters, session_config
from dynamic_map.session import MappingSession
from lidar_interface.lac import LacLog
from lidar_interface.stereo import StereoConfig, stereo_to_points
from pointcloud_processing.preprocessing import PreprocessConfig, preprocess

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"
STEREO = CONFIG / "stereo"


def stereo_config() -> StereoConfig:
    values = load_node_parameters([STEREO / "lac.yaml"], "stereo_cloud")
    fields = StereoConfig.__dataclass_fields__
    return StereoConfig(**{key: value for key, value in values.items() if key in fields})


def preprocess_config() -> PreprocessConfig:
    values = load_node_parameters([CONFIG / "preprocessing.yaml", STEREO / "preprocessing.yaml"], "preprocessing")
    fields = PreprocessConfig.__dataclass_fields__
    return PreprocessConfig(**{key: value for key, value in values.items() if key in fields})


def mapping_values(overrides: list[str]) -> dict:
    files = [CONFIG / name for name in ("mapping.yaml", "features.yaml", "clipper.yaml", "registration.yaml")]
    values = load_node_parameters(files + [STEREO / "mapping.yaml"], "mapping")
    for item in overrides:
        key, raw = item.split("=", 1)
        values[key] = json.loads(raw) if raw[:1] in "0123456789-[{tf" else raw
    return values


def yaw_of(transform: np.ndarray) -> float:
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lac", nargs="?", default=str(REPO / "software" / "pointclouds" / "example (2).lac"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="mapping parameter key=value")
    parser.add_argument("--output", default="/workspace/logs/stereo_benchmark")
    args = parser.parse_args()

    log = LacLog(args.lac)
    stereo = stereo_config()
    pre = preprocess_config()
    values = mapping_values(args.overrides)
    session = MappingSession(session_config(values))
    base_from_camera = np.linalg.inv(session.config.initial_transform)
    pairs = [frame for frame in log.frames if frame.has_stereo]
    origin_inverse = np.linalg.inv(pairs[0].rover)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    previous_truth = None
    for index, frame in enumerate(pairs):
        left, right = log.stereo_pair(frame.frame)
        started = time.perf_counter()
        raw = stereo_to_points(left, right, stereo)
        stereo_s = time.perf_counter() - started
        started = time.perf_counter()
        filtered = preprocess(raw, pre)
        preprocess_s = time.perf_counter() - started
        result = session.process_scan(filtered, frame.mission_time)
        estimate = result.transform @ base_from_camera
        truth = origin_inverse @ frame.rover
        error = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
        step = None if previous_truth is None else float(np.linalg.norm(truth[:3, 3] - previous_truth[:3, 3]))
        previous_truth = truth
        record = {
            "pair": index,
            "frame": frame.frame,
            "mission_time": frame.mission_time,
            "stereo_points": int(raw.shape[0]),
            "processed_points": int(filtered.shape[0]),
            "stereo_s": stereo_s,
            "preprocess_s": preprocess_s,
            "accepted": result.accepted,
            "reason": result.reason,
            "rmse": result.rmse,
            "overlap": result.overlap,
            "inliers": result.inliers,
            "voxels": len(session.map.voxels),
            "pose": estimate[:3, 3].round(4).tolist(),
            "truth": truth[:3, 3].round(4).tolist(),
            "yaw_error_rad": float(np.angle(np.exp(1j * (yaw_of(estimate) - yaw_of(truth))))),
            "translation_error": error,
            "truth_step_m": step,
        }
        record.update({key: float(value) for key, value in result.timings.items()})
        records.append(record)
        print(
            f"pair {index:3d} frame {frame.frame:5d} {result.reason:20s} inliers={result.inliers:3d} "
            f"rmse={result.rmse:6.3f} overlap={result.overlap:4.2f} err={error:5.2f} m "
            f"total={result.timings.get('total_s', 0.0) * 1000:6.1f} ms voxels={len(session.map.voxels)}",
            flush=True,
        )

    errors = np.array([row["translation_error"] for row in records])
    accepted = [row for row in records if row["accepted"]]
    path_length = float(sum(row["truth_step_m"] or 0.0 for row in records))

    def mean(key: str, rows: list[dict] = accepted) -> float | None:
        values_ = [float(row[key]) for row in rows if key in row]
        return float(np.mean(values_)) if values_ else None

    summary = {
        "lac": args.lac,
        "pairs": len(records),
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "rejections": {
            reason: sum(1 for row in records if row["reason"] == reason)
            for reason in sorted({row["reason"] for row in records if not row["accepted"]})
        },
        "path_length_m": path_length,
        "final_pose": records[-1]["pose"],
        "final_truth": records[-1]["truth"],
        "final_translation_error_m": float(errors[-1]),
        "final_error_percent_of_path": float(100.0 * errors[-1] / max(path_length, 1e-9)),
        "max_translation_error_m": float(errors.max()),
        "rmse_translation_error_m": float(np.sqrt(np.mean(errors ** 2))),
        "final_yaw_error_rad": records[-1]["yaw_error_rad"],
        "final_voxels": len(session.map.voxels),
        "mean_stereo_points": mean("stereo_points", records),
        "mean_processed_points": mean("processed_points", records),
        "mean_stereo_s": mean("stereo_s", records),
        "mean_preprocess_s": mean("preprocess_s", records),
        "mean_features_s": mean("features_s"),
        "mean_candidates_s": mean("candidates_s"),
        "mean_clipper_s": mean("clipper_s"),
        "mean_registration_s": mean("registration_s"),
        "mean_map_s": mean("map_s"),
        "mean_session_total_s": mean("total_s", records),
        "parameters": values,
    }
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    output.with_name(output.name + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "parameters"}, indent=2))


if __name__ == "__main__":
    main()
