"""PointCloud2 conversion that does not assume a vendor point layout."""

from __future__ import annotations

import math
from array import array

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from lidar_interface.types import PointCloud

_FIELD_INFO: dict[int, tuple[str, int]] = {
    PointField.INT8: ("i1", 1),
    PointField.UINT8: ("u1", 1),
    PointField.INT16: ("i2", 2),
    PointField.UINT16: ("u2", 2),
    PointField.INT32: ("i4", 4),
    PointField.UINT32: ("u4", 4),
    PointField.FLOAT32: ("f4", 4),
    PointField.FLOAT64: ("f8", 8),
}


def stamp_from_seconds(stamp_sec: float) -> Time:
    """Convert a floating-point second value to a ROS time."""
    if not math.isfinite(stamp_sec):
        raise ValueError("stamp is not finite")
    seconds = math.floor(stamp_sec)
    nanosec = int(round((stamp_sec - seconds) * 1e9))
    if nanosec >= 1_000_000_000:
        seconds += 1
        nanosec -= 1_000_000_000
    if nanosec < 0:
        seconds -= 1
        nanosec += 1_000_000_000
    stamp = Time()
    stamp.sec = int(seconds)
    stamp.nanosec = int(nanosec)
    return stamp


def seconds_from_stamp(stamp: Time) -> float:
    """Convert a ROS time to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _numpy_dtype(datatype: int, endian: str) -> str | None:
    info = _FIELD_INFO.get(int(datatype))
    if info is None:
        return None
    code, _size = info
    if code in ("i1", "u1"):
        return code
    return endian + code


def from_pointcloud2(msg: PointCloud2) -> PointCloud:
    """Convert a PointCloud2 message into xyz coordinates and optional fields.

    x, y, and z are required. Extra scalar fields are preserved when their
    layout fits in ``point_step``. The function does not drop NaN values;
    call :func:`lidar_interface.filters.apply_limits` for that.
    """
    names = [field.name for field in msg.fields]
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"PointCloud2 is missing the '{required}' field")
    if msg.point_step <= 0:
        raise ValueError("PointCloud2 point_step must be positive")
    count = int(msg.width) * int(msg.height)
    raw = bytes(msg.data)
    expected = count * int(msg.point_step)
    if count == 0 or len(raw) == 0:
        return PointCloud(xyz=np.zeros((0, 3)), stamp_sec=seconds_from_stamp(msg.header.stamp), frame_id=msg.header.frame_id)
    if len(raw) < expected:
        raise ValueError(
            f"PointCloud2 data is short: got {len(raw)} bytes, expected {expected}"
        )

    endian = ">" if msg.is_bigendian else "<"
    dtype_fields: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for field in msg.fields:
        if field.name in seen or int(field.count) != 1:
            continue
        numpy_type = _numpy_dtype(int(field.datatype), endian)
        if numpy_type is None:
            continue
        offset = int(field.offset)
        if offset < 0 or offset >= int(msg.point_step):
            raise ValueError(f"Field '{field.name}' offset {offset} is outside the point")
        dtype_fields.append((field.name, numpy_type, offset))
        seen.add(field.name)

    dtype = np.dtype(
        {
            "names": [item[0] for item in dtype_fields],
            "formats": [item[1] for item in dtype_fields],
            "offsets": [item[2] for item in dtype_fields],
            "itemsize": int(msg.point_step),
        }
    )
    records = np.frombuffer(raw[:expected], dtype=dtype, count=count)
    xyz = np.stack(
        [
            np.asarray(records["x"], dtype=np.float64),
            np.asarray(records["y"], dtype=np.float64),
            np.asarray(records["z"], dtype=np.float64),
        ],
        axis=1,
    )
    intensity = None
    label = None
    extras: dict[str, np.ndarray] = {}
    for name in dtype.names:
        if name in ("x", "y", "z"):
            continue
        values = np.asarray(records[name], dtype=np.float64).reshape(-1)
        if name == "intensity":
            intensity = values
        elif name == "label":
            label = values
        else:
            extras[name] = values
    return PointCloud(
        xyz=xyz,
        stamp_sec=seconds_from_stamp(msg.header.stamp),
        frame_id=msg.header.frame_id or "",
        intensity=intensity,
        label=label,
        extras=extras,
    )


def to_pointcloud2(cloud: PointCloud) -> PointCloud2:
    """Pack xyz, and optional intensity and label, into a PointCloud2 message."""
    count = cloud.count
    names = ["x", "y", "z"]
    arrays = [cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]]
    if cloud.intensity is not None:
        if cloud.intensity.shape[0] != count:
            raise ValueError("intensity length does not match xyz")
        names.append("intensity")
        arrays.append(cloud.intensity)
    if cloud.label is not None:
        if cloud.label.shape[0] != count:
            raise ValueError("label length does not match xyz")
        names.append("label")
        arrays.append(cloud.label)

    dtype = np.dtype([(name, "<f4") for name in names])
    packed = np.empty(count, dtype=dtype)
    for name, values in zip(names, arrays):
        packed[name] = np.asarray(values, dtype=np.float32)

    msg = PointCloud2()
    header = Header()
    header.stamp = stamp_from_seconds(cloud.stamp_sec)
    header.frame_id = cloud.frame_id
    msg.header = header
    msg.height = 1
    msg.width = count
    msg.fields = []
    offset = 0
    for name in names:
        field = PointField()
        field.name = name
        field.offset = offset
        field.datatype = PointField.FLOAT32
        field.count = 1
        msg.fields.append(field)
        offset += 4
    msg.is_bigendian = False
    msg.point_step = offset
    msg.row_step = offset * count
    msg.is_dense = bool(np.isfinite(cloud.xyz).all()) if count else True
    msg.data = array("B", packed.tobytes())
    return msg
