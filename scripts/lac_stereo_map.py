#!/usr/bin/env python3
"""Build a ground-truth-posed stereo map from a Lunar Autonomy Challenge .lac log.

Each FrontLeft/FrontRight pair goes through the same SGBM code as the
``stereo_cloud`` node. Points are placed in the world with the logged rover
pose and the documented camera mount, then fused into the voxel map.

The consistency check registers nothing: it measures how far each frame's
world points lie from the previous frame's. A wrong focal length or baseline
scales depth and shows up as a large distance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from dynamic_map.voxel_map import MapConfig, VoxelMap
from lidar_interface.stereo import StereoConfig, stereo_to_points

FRONT_LEFT_MOUNT = np.array([0.28, 0.081, 0.131])


def rover_transform(row) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [row["roll"], row["pitch"], row["yaw"]]).as_matrix()
    transform[:3, 3] = [row["x"], row["y"], row["z"]]
    return transform


def top_down_png(points: np.ndarray, path: Path, resolution: float = 0.05) -> None:
    from PIL import Image

    low = points[:, :2].min(axis=0)
    size = np.ceil((points[:, :2].max(axis=0) - low) / resolution).astype(int) + 1
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cols = ((points[:, 0] - low[0]) / resolution).astype(int)
    rows = size[1] - 1 - ((points[:, 1] - low[1]) / resolution).astype(int)
    z = points[:, 2]
    span = np.percentile(z, [2, 98])
    weight = np.clip((z - span[0]) / max(span[1] - span[0], 1e-6), 0.0, 1.0)
    colors = np.stack([255 * weight, 80 + 120 * (1 - np.abs(weight - 0.5) * 2), 255 * (1 - weight)], axis=1)
    order = np.argsort(z)
    image[rows[order], cols[order]] = colors[order].astype(np.uint8)
    Image.fromarray(image).resize((size[0] * 2, size[1] * 2), Image.NEAREST).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lac", help="path to the .lac file")
    parser.add_argument("--output", default="/workspace/logs/lac_stereo")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=6.0)
    args = parser.parse_args()

    from lunarloc import CameraDataReader, FrameDataReader

    frames = FrameDataReader(args.lac).frames
    cameras = CameraDataReader(args.lac)
    config = StereoConfig(scale=args.scale, max_depth=args.max_depth)
    voxel_map = VoxelMap(MapConfig(resolution=0.1, min_static_observations=2, decay_lambda=0.0, max_voxels=500_000))
    previous = None
    records = []
    for _index, row in frames.iterrows():
        frame = int(row["frame"])
        left = cameras.get_image("FrontLeft", frame, "grayscale")
        right = cameras.get_image("FrontRight", frame, "grayscale")
        if left is None or right is None:
            continue
        started = time.perf_counter()
        local = stereo_to_points(np.asarray(left), np.asarray(right), config)
        stereo_s = time.perf_counter() - started
        camera = rover_transform(row)
        camera[:3, 3] = camera[:3, :3] @ FRONT_LEFT_MOUNT + camera[:3, 3]
        world = local @ camera[:3, :3].T + camera[:3, 3]
        fused_at = time.perf_counter()
        voxel_map.integrate_points(world, camera[:3, 3], float(row["mission_time"]))
        map_s = time.perf_counter() - fused_at
        record = {
            "frame": frame,
            "points": int(local.shape[0]),
            "stereo_s": stereo_s,
            "map_s": map_s,
            "median_depth_m": float(np.median(local[:, 0])) if local.shape[0] else None,
            "camera_height_above_points_m": float(np.median(-local[:, 2])) if local.shape[0] else None,
        }
        if previous is not None and local.shape[0] and previous.shape[0]:
            distance, _ = cKDTree(previous).query(world, k=1)
            record["median_distance_to_previous_m"] = float(np.median(distance))
        previous = world
        records.append(record)
        print(json.dumps(record), flush=True)

    cloud = voxel_map.export(float(frames["mission_time"].iloc[-1]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d

        ply = o3d.geometry.PointCloud()
        ply.points = o3d.utility.Vector3dVector(cloud.xyz)
        o3d.io.write_point_cloud(str(output.with_suffix(".ply")), ply)
    except ImportError:
        pass
    if cloud.xyz.shape[0]:
        top_down_png(cloud.xyz, output.with_suffix(".png"))
    distances = [r["median_distance_to_previous_m"] for r in records if "median_distance_to_previous_m" in r]
    summary = {
        "image_pairs": len(records),
        "mean_points_per_pair": float(np.mean([r["points"] for r in records])) if records else 0.0,
        "mean_stereo_s": float(np.mean([r["stereo_s"] for r in records])) if records else None,
        "mean_map_s": float(np.mean([r["map_s"] for r in records])) if records else None,
        "median_depth_m": float(np.median([r["median_depth_m"] for r in records if r["median_depth_m"] is not None])),
        "median_camera_height_above_points_m": float(
            np.median([r["camera_height_above_points_m"] for r in records if r["camera_height_above_points_m"] is not None])
        ),
        "median_frame_to_frame_distance_m": float(np.median(distances)) if distances else None,
        "map_voxels": int(cloud.xyz.shape[0]),
        "map_extent_m": (cloud.xyz.max(axis=0) - cloud.xyz.min(axis=0)).round(2).tolist() if cloud.xyz.shape[0] else None,
        "config": config.__dict__,
    }
    output.with_name(output.name + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
