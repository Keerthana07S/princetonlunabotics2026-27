"""Small rigid-transform helpers. Matrices map column vectors as ``p' = T p``."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to ``(N, 3)`` points."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return (rotation @ xyz.T).T + translation


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""
    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ matrix[:3, 3])
    return inverse


def yaw_rotation(yaw_rad: float) -> np.ndarray:
    """Rotation matrix for a yaw about +z, following REP 103."""
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 matrix from a 3x3 rotation and a length-3 translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Return ``(x, y, z, w)`` for a 3x3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    return Rotation.from_matrix(matrix).as_quat()
