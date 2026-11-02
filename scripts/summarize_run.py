#!/usr/bin/env python3
"""Summarize one live launch from the per-node logs/metrics_*.jsonl files.

Only rows written after ``--since`` (Unix wall time) are used. Tracking error
compares each accepted mapping pose with the synthetic ground truth of the scan
with the same stamp, both expressed relative to the first mapped scan. Every
number printed comes from the log; nothing is estimated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def rate(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    times = sorted(float(row["wall_time"]) for row in rows)
    return (len(times) - 1) / max(times[-1] - times[0], 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-glob", default="/workspace/logs/metrics_*.jsonl")
    parser.add_argument("--since", type=float, required=True, help="Unix time the launch started")
    parser.add_argument("--output", default="/workspace/logs/e2e_summary.json")
    args = parser.parse_args()

    rows = []
    undecodable = 0
    for path in sorted(Path("/").glob(args.metrics_glob.lstrip("/"))):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                undecodable += 1
                continue
            if float(row.get("wall_time", 0.0)) >= args.since:
                rows.append(row)
    by_stage: dict[str, list[dict]] = {}
    for row in rows:
        by_stage.setdefault(row["stage"], []).append(row)
    scans = by_stage.get("synthetic_scan", [])
    pre = by_stage.get("preprocessing", [])
    mapped = by_stage.get("mapping", [])
    rejected = by_stage.get("registration_rejected", [])

    truth = {round(float(row["stamp"]), 6): row for row in scans if "stamp" in row}
    errors = []
    final = None
    origin = None
    for row in sorted(mapped, key=lambda item: float(item["stamp"])):
        match = truth.get(round(float(row["stamp"]), 6))
        if match is None:
            continue
        if origin is None:
            origin = match
        yaw = float(origin["truth_yaw"])
        offset = np.array(
            [
                float(match["truth_x"]) - float(origin["truth_x"]),
                float(match["truth_y"]) - float(origin["truth_y"]),
                float(match["truth_z"]) - float(origin["truth_z"]),
            ]
        )
        cos, sin = math.cos(-yaw), math.sin(-yaw)
        truth_map = np.array([cos * offset[0] - sin * offset[1], sin * offset[0] + cos * offset[1], offset[2]])
        estimate = np.array([float(row["pose_x"]), float(row["pose_y"]), float(row["pose_z"])])
        errors.append(float(np.linalg.norm(estimate - truth_map)))
        final = {"estimate": estimate.round(4).tolist(), "truth": truth_map.round(4).tolist()}

    moving = [row for row in scans if float(row.get("scene_time", 0.0)) > 0.0]
    summary = {
        "undecodable_metric_lines": undecodable,
        "window_s": (max(float(r["wall_time"]) for r in rows) - min(float(r["wall_time"]) for r in rows)) if rows else 0,
        "synthetic_scans": len(scans),
        "synthetic_scans_while_driving": len(moving),
        "preprocessed_scans": len(pre),
        "mapping_accepted": len(mapped),
        "mapping_rejected": len(rejected),
        "rejection_reasons": sorted({str(row.get("reason")) for row in rejected}),
        "scan_rate_hz": rate(scans),
        "preprocess_rate_hz": rate(pre),
        "accepted_map_update_rate_hz": rate(mapped),
        "scan_to_map_latency_s": stats([float(row["latency_s"]) for row in mapped if "latency_s" in row]),
        "mapping_callback_s": stats([float(row["duration_s"]) for row in mapped]),
        "preprocess_callback_s": stats([float(row["duration_s"]) for row in pre]),
        "synthetic_scan_generation_s": stats([float(row["duration_s"]) for row in scans]),
        "features_s": stats([float(row["features_s"]) for row in mapped if "features_s" in row]),
        "clipper_s": stats([float(row["clipper_s"]) for row in mapped if "clipper_s" in row]),
        "registration_s": stats([float(row["registration_s"]) for row in mapped if "registration_s" in row]),
        "map_update_s": stats([float(row["map_s"]) for row in mapped if "map_s" in row]),
        "input_points": stats([float(row["points"]) for row in scans]),
        "processed_points": stats([float(row["output_points"]) for row in pre]),
        "final_voxels": int(mapped[-1]["voxels"]) if mapped else None,
        "mapping_rss_mb": stats([float(row["rss_mb"]) for row in mapped if float(row.get("rss_mb", -1)) > 0]),
        "translation_error_m": stats(errors),
        "final_translation_error_m": errors[-1] if errors else None,
        "final_pose": final,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
