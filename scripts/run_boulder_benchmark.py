#!/usr/bin/env python3
"""Associate boulder detections from a traverse CSV to a landmark map with CLIPPER.

The logic is ``pointcloud_processing.boulder_association``; see its docstring
for the frames, odometry model, and ground-truth identities. This script runs
a whole traverse and writes per-frame records and a summary. To watch a run in
Foxglove, use ``ros2 launch mapping_visualizer boulder_clipper.launch.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_map.params import load_node_parameters
from pointcloud_processing.boulder_association import BoulderConfig, BoulderSession, load_traverse
from pointcloud_processing.clipper_association import ClipperConfig

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"


def main() -> None:
    defaults = BoulderConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--scale-error", type=float, default=defaults.scale_error, help="odometry distance scale error")
    parser.add_argument("--translation-noise", type=float, default=defaults.translation_noise, help="std per meter")
    parser.add_argument("--yaw-drift-deg-per-m", type=float, default=defaults.yaw_drift_deg_per_m)
    parser.add_argument("--yaw-noise-deg-per-m", type=float, default=defaults.yaw_noise_deg_per_m)
    parser.add_argument("--identity-radius", type=float, default=defaults.identity_radius)
    parser.add_argument("--gate-radius", type=float, default=defaults.gate_radius)
    parser.add_argument("--max-candidates-per-detection", type=int, default=defaults.max_candidates_per_detection)
    parser.add_argument("--min-inliers", type=int, default=defaults.clipper.min_inliers)
    parser.add_argument("--max-rmse", type=float, default=defaults.max_rmse)
    parser.add_argument("--max-correction-yaw-deg", type=float, default=defaults.max_correction_yaw_deg)
    parser.add_argument("--max-correction-m", type=float, default=defaults.max_correction_m)
    parser.add_argument("--merge-radius", type=float, default=defaults.merge_radius)
    parser.add_argument("--sigma", type=float, default=defaults.clipper.sigma, help="negative uses config/clipper.yaml")
    parser.add_argument("--epsilon", type=float, default=defaults.clipper.epsilon, help="negative uses config/clipper.yaml")
    parser.add_argument("--no-correction", action="store_true", help="run odometry only, CLIPPER still scored")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    clipper_values = load_node_parameters([CONFIG / "clipper.yaml"], "mapping")
    config = BoulderConfig(
        seed=args.seed,
        scale_error=args.scale_error,
        translation_noise=args.translation_noise,
        yaw_drift_deg_per_m=args.yaw_drift_deg_per_m,
        yaw_noise_deg_per_m=args.yaw_noise_deg_per_m,
        identity_radius=args.identity_radius,
        gate_radius=args.gate_radius,
        max_candidates_per_detection=args.max_candidates_per_detection,
        max_rmse=args.max_rmse,
        max_correction_yaw_deg=args.max_correction_yaw_deg,
        max_correction_m=args.max_correction_m,
        merge_radius=args.merge_radius,
        correct=not args.no_correction,
        clipper=ClipperConfig(
            sigma=float(clipper_values["clipper_sigma"] if args.sigma < 0 else args.sigma),
            epsilon=float(clipper_values["clipper_epsilon"] if args.epsilon < 0 else args.epsilon),
            mindist=float(clipper_values["clipper_mindist"]),
            min_inliers=args.min_inliers,
        ),
    )
    output = args.output or REPO / "logs" / f"boulder_benchmark_{args.csv.stem.replace(' ', '_')}"
    output.parent.mkdir(parents=True, exist_ok=True)

    session = BoulderSession(load_traverse(args.csv), config)
    while (result := session.step()) is not None:
        row = result.record
        print(
            f"frame {row['frame']:6d} det={row['detections']:2d} cand={row['candidates']:3d} "
            f"inl={row['clipper_inliers']:2d} {row['reason']:18s} err={row['translation_error']:6.3f} "
            f"odom={row['odometry_translation_error']:6.3f} yaw={row['yaw_deg']:6.2f} "
            f"odom_yaw={row['odometry_yaw_deg']:6.2f} landmarks={row['landmarks']}",
            flush=True,
        )

    summary = {"csv": str(args.csv), **session.summary()}
    summary["parameters"] = {
        **{key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "clipper_sigma": config.clipper.sigma,
        "clipper_epsilon": config.clipper.epsilon,
        "clipper_mindist": config.clipper.mindist,
    }
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in session.records:
            handle.write(json.dumps(row) + "\n")
    output.with_name(output.name + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "parameters"}, indent=2))


if __name__ == "__main__":
    main()
