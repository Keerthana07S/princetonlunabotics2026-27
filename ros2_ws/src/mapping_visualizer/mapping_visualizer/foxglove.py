"""Messages that let Foxglove show what the Viser process draws itself."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from lidar_interface.types import PointCloud
from mapping_visualizer.colors import confidence_colors


def pack_rgb(colors: np.ndarray) -> np.ndarray:
    """Pack ``(N, 3)`` colors in [0, 1] as PCL-style ``0x00RRGGBB`` uint32, which Foxglove reads as "BGR (packed)"."""
    channels = np.clip(np.rint(np.asarray(colors, dtype=np.float64).reshape(-1, 3) * 255.0), 0, 255).astype(np.uint32)
    return (channels[:, 0] << 16) | (channels[:, 1] << 8) | channels[:, 2]


def colored_map(cloud: PointCloud, header: Header, low_confidence: float, high_confidence: float) -> PointCloud2:
    """Map cloud with x, y, z, packed ``rgb`` (confidence colors, dynamic in orange), ``confidence``, ``dynamic``, and ``age`` if present."""
    count = int(cloud.count)
    confidence = np.ones(count) if cloud.intensity is None else np.asarray(cloud.intensity, dtype=np.float64)
    dynamic = np.zeros(count) if cloud.label is None else np.asarray(cloud.label, dtype=np.float64)
    columns = [
        ("x", PointField.FLOAT32, "<f4", cloud.xyz[:, 0]),
        ("y", PointField.FLOAT32, "<f4", cloud.xyz[:, 1]),
        ("z", PointField.FLOAT32, "<f4", cloud.xyz[:, 2]),
        ("rgb", PointField.UINT32, "<u4", pack_rgb(confidence_colors(confidence, dynamic > 0.5, low_confidence, high_confidence))),
        ("confidence", PointField.FLOAT32, "<f4", confidence),
        ("dynamic", PointField.FLOAT32, "<f4", dynamic),
    ]
    age = None if cloud.extras is None else cloud.extras.get("age")
    if age is not None:
        columns.append(("age", PointField.FLOAT32, "<f4", age))
    dtype = np.dtype([(name, code) for name, _datatype, code, _values in columns])
    data = np.zeros(count, dtype=dtype)
    message = PointCloud2()
    message.header = header
    message.height = 1
    message.width = count
    message.is_bigendian = False
    message.is_dense = True
    message.point_step = dtype.itemsize
    message.row_step = dtype.itemsize * count
    for name, datatype, _code, values in columns:
        data[name] = values
        field = PointField()
        field.name = name
        field.offset = dtype.fields[name][1]
        field.datatype = datatype
        field.count = 1
        message.fields.append(field)
    message.data = data.tobytes()
    return message


class RelativePose:
    """Express poses relative to the first one received, which is where the map frame starts."""

    def __init__(self) -> None:
        self._origin: tuple[np.ndarray, Rotation] | None = None

    def __call__(self, position: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, Rotation]:
        if self._origin is None:
            self._origin = (np.asarray(position, dtype=np.float64), rotation)
        origin, origin_rotation = self._origin
        inverse = origin_rotation.inv()
        return inverse.apply(np.asarray(position, dtype=np.float64) - origin), inverse * rotation


class PathRecorder:
    """``nav_msgs/Path`` that keeps a pose only after moving ``spacing`` metres, up to ``max_poses``."""

    def __init__(self, frame_id: str, spacing: float, max_poses: int) -> None:
        self.path = Path()
        self.path.header.frame_id = frame_id
        self._spacing = float(spacing)
        self._max_poses = int(max_poses)
        self._last: np.ndarray | None = None

    def add(self, pose: PoseStamped) -> bool:
        """Return True if ``pose`` was kept."""
        position = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        if self._last is not None and np.linalg.norm(position - self._last) < self._spacing:
            return False
        self._last = position
        self.path.poses.append(pose)
        if len(self.path.poses) > self._max_poses:
            del self.path.poses[: len(self.path.poses) - self._max_poses]
        self.path.header.stamp = pose.header.stamp
        return True
