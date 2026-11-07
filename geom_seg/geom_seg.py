"""
Minimal geometric (unsupervised) point cloud segmentation.

Pipeline:
  1. Voxel downsample       -> fewer points, faster everything downstream
  2. RANSAC plane removal   -> strip ground / walls (large planar surfaces)
  3. DBSCAN clustering      -> remaining points grouped into candidate objects
  4. Inverse projection     -> labels pushed back onto the full-resolution cloud
  5. Per-object boxes       -> centroid, size, distance (input for avoidance)

Usage:
  python geom_seg.py                  # runs on a synthetic scene
  python geom_seg.py scan.ply         # any file Open3D can read (ply/pcd/xyz...)
  python geom_seg.py scan.ply --viz   # open a viewer window
  python csv_to_pcd.py scan.csv scan.pcd  # convert CSV XYZ data first

Deps: pip install numpy open3d scipy
"""
import argparse
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ----------------------------------------------------------------- core steps
def remove_planes(pcd, dist_thresh=0.03, max_planes=3, min_plane_frac=0.05):
    """Iteratively RANSAC-fit planes and strip them. Returns (objects, planes).

    Stops when the best plane holds < min_plane_frac of the ORIGINAL points,
    so small flat faces of real objects (e.g. a box top) are left alone.
    """
    n_total = len(pcd.points)
    remaining, planes = pcd, []
    for _ in range(max_planes):
        if len(remaining.points) < 3:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=500
        )
        if len(inliers) < min_plane_frac * n_total:
            break
        planes.append((model, remaining.select_by_index(inliers)))
        remaining = remaining.select_by_index(inliers, invert=True)
    return remaining, planes

def cluster_objects(pcd, eps=0.15, min_points=20):
    """DBSCAN on the non-planar points. Label -1 = noise."""
    if len(pcd.points) == 0:
        return np.empty(0, dtype=int)
    return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))

def propagate_labels(src_xyz, src_labels, dst_xyz, max_dist):
    """Inverse projection: give every full-res point the label of its nearest
    downsampled point (or -1 if nothing is within max_dist)."""
    if len(src_xyz) == 0:
        return np.full(len(dst_xyz), -1, dtype=int)
    dist, idx = cKDTree(src_xyz).query(dst_xyz, k=1, distance_upper_bound=max_dist)
    out = np.full(len(dst_xyz), -1, dtype=int)
    ok = np.isfinite(dist)
    out[ok] = src_labels[idx[ok]]
    return out

def describe_objects(xyz, labels, sensor_origin=(0.0, 0.0, 0.0)):
    """Axis-aligned box + range for each cluster, nearest first."""
    origin = np.asarray(sensor_origin)
    objs = []
    for lab in np.unique(labels[labels >= 0]):
        pts = xyz[labels == lab]
        lo, hi = pts.min(0), pts.max(0)
        c = pts.mean(0)
        objs.append(
            dict(
                id=int(lab),
                n_points=len(pts),
                centroid=c,
                size=hi - lo,
                min=lo,
                max=hi,
                distance=float(np.linalg.norm(c - origin)),
            )
        )
    return sorted(objs, key=lambda o: o["distance"])

# --------------------------------------------------------------- full pipeline
def segment(pcd, voxel=0.05, plane_thresh=0.03, eps=None, min_points=15, max_planes=3):
    if voxel <= 0 or plane_thresh <= 0 or min_points < 1 or max_planes < 0:
        raise ValueError("voxel and plane_thresh must be positive; min_points must be at least 1")

    full_xyz = np.asarray(pcd.points)
    finite = np.isfinite(full_xyz).all(axis=1)
    nonzero = np.linalg.norm(full_xyz, axis=1) > np.finfo(float).eps
    valid = finite & nonzero
    if not np.any(valid):
        raise ValueError("point cloud contains no finite, non-zero XYZ points")

    # Stereo depth clouds commonly contain NaN/Inf or zero points for pixels
    # without depth. Keep those placeholders out of processing and rendering.
    work_pcd = pcd.select_by_index(np.flatnonzero(valid))
    eps = eps or voxel * 3  # clustering radius tied to point spacing
    t0 = time.perf_counter()

    down = work_pcd.voxel_down_sample(voxel)
    objects_pcd, planes = remove_planes(down, plane_thresh, max_planes)

    # Cluster the non-planar points, then re-attach to the downsampled cloud.
    obj_labels = cluster_objects(objects_pcd, eps, min_points)
    obj_xyz = np.asarray(objects_pcd.points)

    valid_xyz = full_xyz[valid]
    valid_labels = propagate_labels(obj_xyz, obj_labels, valid_xyz, max_dist=voxel * 1.5)

    # Mark plane points (ground etc.) as -2 so they're distinguishable from noise.
    for _, plane_pcd in planes:
        plane_xyz = np.asarray(plane_pcd.points)
        if len(plane_xyz) == 0:
            continue
        d, _ = cKDTree(plane_xyz).query(valid_xyz, k=1)
        near_plane = (d < voxel * 0.75) & (valid_labels == -1)
        valid_labels[near_plane] = -2

    labels = np.full(len(full_xyz), -1, dtype=int)
    labels[valid] = valid_labels
    objs = describe_objects(valid_xyz, valid_labels)
    elapsed = time.perf_counter() - t0
    return labels, objs, planes, elapsed

# ---------------------------------------------------------------- demo / viz
def synthetic_scene(n_ground=200_000, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    # ground plane (z ~ 0)
    g = rng.uniform([-10, -10, 0], [10, 10, 0], (n_ground, 3))
    g[:, 2] += rng.normal(0, 0.005, n_ground)
    parts.append(g)
    # two boxes
    for (cx, cy, s) in [(3, 2, 1.0), (-4, -1, 1.5)]:
        b = rng.uniform([cx - s / 2, cy - s / 2, 0], [cx + s / 2, cy + s / 2, s], (30_000, 3))
        face = rng.integers(0, 3, len(b))
        side = rng.integers(0, 2, len(b))
        lo = np.array([cx - s / 2, cy - s / 2, 0.0])
        hi = np.array([cx + s / 2, cy + s / 2, s])
        b[np.arange(len(b)), face] = np.where(side, hi[face], lo[face])  # surface only
        parts.append(b)
    # a cylinder-ish pole
    th = rng.uniform(0, 2 * np.pi, 15_000)
    parts.append(np.c_[1 + 0.15 * np.cos(th), -3 + 0.15 * np.sin(th), rng.uniform(0, 2, 15_000)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(parts))
    return pcd

def colorize(pcd, labels):
    if len(labels) != len(pcd.points):
        raise ValueError("labels must contain one entry per point")
    rng = np.random.default_rng(1)
    max_label = int(labels.max()) if len(labels) else -1
    palette = rng.uniform(0.2, 1.0, (max(max_label + 1, 1), 3))
    colors = np.full((len(labels), 3), 0.5)  # noise = gray
    colors[labels == -2] = (0.25, 0.25, 0.25)  # planes = dark gray
    m = labels >= 0
    colors[m] = palette[labels[m]]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize(pcd, labels):
    """Open an interactive viewer for the segmented point cloud."""
    xyz = np.asarray(pcd.points)
    valid = np.isfinite(xyz).all(axis=1)
    valid &= np.linalg.norm(xyz, axis=1) > np.finfo(float).eps
    if not np.any(valid):
        raise ValueError("point cloud contains no points to visualize")

    display_pcd = pcd.select_by_index(np.flatnonzero(valid))
    display_pcd = colorize(display_pcd, labels[valid])
    extent = np.ptp(np.asarray(display_pcd.points), axis=0).max()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(float(extent) * 0.1, 0.05)
    )

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="Geometric Segmentation")
    viewer.add_geometry(display_pcd)
    viewer.add_geometry(frame)
    options = viewer.get_render_option()
    options.background_color = np.array([0.05, 0.05, 0.05])
    options.point_size = 2.0
    viewer.run()
    viewer.destroy_window()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="point cloud file (default: synthetic demo)")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    if args.path:
        pcd = o3d.io.read_point_cloud(args.path)
        if pcd.is_empty():
            raise SystemExit(f"could not read any points from {args.path!r}")
    else:
        pcd = synthetic_scene()
    print(f"loaded {len(pcd.points):,} points")

    labels, objs, planes, dt = segment(pcd, voxel=args.voxel)
    print(f"segmented in {dt:.2f}s  |  {len(planes)} plane(s) removed  |  {len(objs)} object(s)")
    for o in objs:
        print(
            f"  obj {o['id']:>2}: {o['n_points']:>6,} pts  "
            f"center={np.round(o['centroid'], 2)}  size={np.round(o['size'], 2)}  "
            f"dist={o['distance']:.2f} m"
        )

    if args.viz:
        visualize(pcd, labels)

if __name__ == "__main__":
    main()