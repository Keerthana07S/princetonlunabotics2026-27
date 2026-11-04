"""PointCloud2 layout tests. Skipped when ROS messages are not installed."""

from __future__ import annotations

from array import array

import numpy as np
import pytest

pytest.importorskip("sensor_msgs")

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from lidar_interface.pointcloud import from_pointcloud2, stamp_from_seconds, to_pointcloud2
from lidar_interface.types import PointCloud


def _manual_cloud(points: np.ndarray, datatype: int, with_intensity: bool) -> PointCloud2:
    count = points.shape[0]
    point_size = 8 if datatype == PointField.FLOAT64 else 4
    names = ["x", "y", "z"] + (["intensity"] if with_intensity else [])
    step = point_size * len(names)
    dtype_char = "<f8" if datatype == PointField.FLOAT64 else "<f4"
    packed = np.zeros(count, dtype=[(name, dtype_char) for name in names])
    for index, name in enumerate(("x", "y", "z")):
        packed[name] = points[:, index]
    if with_intensity:
        packed["intensity"] = np.arange(count, dtype=np.float64)
    message = PointCloud2()
    message.header = Header()
    message.header.stamp = stamp_from_seconds(12.5)
    message.header.frame_id = "lidar_link"
    message.height = 1
    message.width = count
    message.point_step = step
    message.row_step = step * count
    message.is_bigendian = False
    message.is_dense = False
    message.fields = []
    for index, name in enumerate(names):
        field = PointField()
        field.name = name
        field.offset = index * point_size
        field.datatype = datatype
        field.count = 1
        message.fields.append(field)
    message.data = array("B", packed.tobytes())
    return message


def test_roundtrip_preserves_xyz_stamp_and_frame() -> None:
    original = PointCloud(
        xyz=np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]),
        stamp_sec=3.25,
        frame_id="lidar_link",
        intensity=np.array([0.1, 0.8]),
        label=np.array([0.0, 2.0]),
    )
    restored = from_pointcloud2(to_pointcloud2(original))
    assert restored.frame_id == "lidar_link"
    assert restored.stamp_sec == pytest.approx(3.25)
    assert np.allclose(restored.xyz, original.xyz, atol=1e-6)
    assert restored.intensity is not None
    assert np.allclose(restored.intensity, original.intensity, atol=1e-6)
    assert restored.label is not None
    assert np.allclose(restored.label, original.label, atol=1e-5)


def test_roundtrip_preserves_extra_fields() -> None:
    original = PointCloud(
        xyz=np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]),
        stamp_sec=1.0,
        frame_id="map",
        intensity=np.array([0.5, 1.5]),
        extras={"age": np.array([0.0, 12.5])},
    )
    message = to_pointcloud2(original)
    assert [field.name for field in message.fields] == ["x", "y", "z", "intensity", "age"]
    restored = from_pointcloud2(message)
    assert np.allclose(restored.extras["age"], [0.0, 12.5])


def test_float64_layout_with_intensity() -> None:
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    restored = from_pointcloud2(_manual_cloud(points, PointField.FLOAT64, with_intensity=True))
    assert np.allclose(restored.xyz, points)
    assert restored.intensity is not None
    assert np.allclose(restored.intensity, [0.0, 1.0])


def test_missing_axis_raises() -> None:
    message = _manual_cloud(np.zeros((1, 3)), PointField.FLOAT32, with_intensity=False)
    message.fields = [field for field in message.fields if field.name != "z"]
    with pytest.raises(ValueError, match="missing"):
        from_pointcloud2(message)


def test_short_buffer_raises() -> None:
    message = _manual_cloud(np.zeros((2, 3)), PointField.FLOAT32, with_intensity=False)
    message.data = message.data[:4]
    with pytest.raises(ValueError, match="short"):
        from_pointcloud2(message)


def test_empty_cloud_roundtrip() -> None:
    cloud = PointCloud(xyz=np.zeros((0, 3)), stamp_sec=0.0, frame_id="lidar_link")
    restored = from_pointcloud2(to_pointcloud2(cloud))
    assert restored.count == 0
    assert restored.frame_id == "lidar_link"
