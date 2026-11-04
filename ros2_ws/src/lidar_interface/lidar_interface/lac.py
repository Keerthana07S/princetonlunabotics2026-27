"""Lunar Autonomy Challenge (.lac) log access without ROS.

A ``.lac`` file is a gzipped tar read with MIT's ``lunarloc`` package. Rover
poses are world-frame ``x, y, z, roll, pitch, yaw`` (extrinsic xyz Euler, i.e.
``R = Rz(yaw) Ry(pitch) Rx(roll)``). Camera mounts are logged per image frame
in the rover body frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class LacFrame:
    """One row of the log. ``has_stereo`` is true when both front images exist."""

    frame: int
    mission_time: float
    rover: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    has_stereo: bool


class LacLog:
    """Rows, stereo images, and the left-camera mount of one recording."""

    def __init__(self, path: str, left: str = "FrontLeft", right: str = "FrontRight") -> None:
        from lunarloc import CameraDataReader, FrameDataReader

        self.left = left
        self.right = right
        reader = FrameDataReader(path)
        self._cameras = CameraDataReader(path)
        left_rows = reader.camera_frames.get(left)
        right_rows = reader.camera_frames.get(right)
        stereo = set()
        if left_rows is not None and right_rows is not None:
            stereo = set(left_rows["frame"].astype(int)) & set(right_rows["frame"].astype(int))
        self.frames = [
            LacFrame(
                frame=int(row["frame"]),
                mission_time=float(row["mission_time"]),
                rover=rover_transform(row),
                accel=np.array([row["accel_x"], row["accel_y"], row["accel_z"]], dtype=np.float64),
                gyro=np.array([row["gyro_x"], row["gyro_y"], row["gyro_z"]], dtype=np.float64),
                has_stereo=int(row["frame"]) in stereo,
            )
            for _index, row in reader.frames.iterrows()
        ]
        self.left_mount = _mount(left_rows)

    def stereo_pair(self, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
        left = self._cameras.get_image(self.left, frame, "grayscale")
        right = self._cameras.get_image(self.right, frame, "grayscale")
        if left is None or right is None:
            return None
        return np.asarray(left, dtype=np.uint8), np.asarray(right, dtype=np.uint8)


def rover_transform(row) -> np.ndarray:
    """World-from-rover 4x4 matrix for one log row."""
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [row["roll"], row["pitch"], row["yaw"]]).as_matrix()
    transform[:3, 3] = [row["x"], row["y"], row["z"]]
    return transform


def _mount(rows) -> np.ndarray:
    """Rover-from-camera 4x4 matrix from the first logged mount; identity when absent."""
    transform = np.eye(4)
    if rows is None or len(rows) == 0:
        return transform
    first = rows.iloc[0]
    transform[:3, :3] = Rotation.from_euler(
        "xyz", [first["camera_roll"], first["camera_pitch"], first["camera_yaw"]]
    ).as_matrix()
    transform[:3, 3] = [first["camera_x"], first["camera_y"], first["camera_z"]]
    return transform
