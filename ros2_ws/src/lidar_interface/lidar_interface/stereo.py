"""Rectified stereo pair to a point cloud with OpenCV semi-global matching.

The output frame is the left camera's body frame (REP 103: x forward, y left,
z up), with its origin at the left camera centre. That makes the cloud a
drop-in LiDAR-like input whose rays start at the sensor origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StereoConfig:
    """Pinhole stereo settings.

    ``horizontal_fov_rad`` and ``baseline`` default to the Lunar Autonomy
    Challenge front pair (1.22 rad, 0.162 m, square pixels, no distortion).
    ``scale`` resizes both images before matching, so ``num_disparities`` is
    in scaled pixels.
    """

    horizontal_fov_rad: float = 1.22
    baseline: float = 0.162
    scale: float = 0.5
    num_disparities: int = 96
    block_size: int = 7
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    min_depth: float = 0.5
    max_depth: float = 8.0
    min_intensity: int = 8
    stride: int = 2


def focal_length(width: int, horizontal_fov_rad: float) -> float:
    """Focal length in pixels for an image ``width`` pixels wide."""
    return 0.5 * width / math.tan(0.5 * horizontal_fov_rad)


def compute_disparity(left: np.ndarray, right: np.ndarray, config: StereoConfig) -> np.ndarray:
    """Disparity in scaled pixels; invalid pixels are ``nan``."""
    import cv2

    left = _gray(left)
    right = _gray(right)
    if left.shape != right.shape:
        raise ValueError(f"stereo images differ in size: {left.shape} vs {right.shape}")
    if config.scale != 1.0:
        size = (max(1, int(round(left.shape[1] * config.scale))), max(1, int(round(left.shape[0] * config.scale))))
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    disparities = max(16, int(math.ceil(config.num_disparities / 16.0)) * 16)
    block = max(3, int(config.block_size) | 1)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=disparities,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        uniquenessRatio=int(config.uniqueness_ratio),
        speckleWindowSize=int(config.speckle_window),
        speckleRange=int(config.speckle_range),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan
    disparity[left < config.min_intensity] = np.nan
    return disparity


def disparity_to_points(disparity: np.ndarray, full_width: int, config: StereoConfig) -> np.ndarray:
    """Back-project a disparity image to ``(N, 3)`` points in the left camera body frame."""
    height, width = disparity.shape
    fx = focal_length(full_width, config.horizontal_fov_rad) * (width / float(full_width))
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    step = max(1, int(config.stride))
    rows, cols = np.mgrid[0:height:step, 0:width:step]
    sampled = disparity[rows, cols]
    valid = np.isfinite(sampled)
    depth = np.full(sampled.shape, np.nan, dtype=np.float64)
    depth[valid] = fx * config.baseline / sampled[valid]
    keep = valid & (depth >= config.min_depth) & (depth <= config.max_depth)
    z_optical = depth[keep]
    x_optical = (cols[keep] - cx) * z_optical / fx
    y_optical = (rows[keep] - cy) * z_optical / fx
    return np.stack([z_optical, -x_optical, -y_optical], axis=1)


def stereo_to_points(left: np.ndarray, right: np.ndarray, config: StereoConfig) -> np.ndarray:
    """Match a rectified pair and return points in the left camera body frame."""
    disparity = compute_disparity(left, right, config)
    return disparity_to_points(disparity, _gray(left).shape[1], config)


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
