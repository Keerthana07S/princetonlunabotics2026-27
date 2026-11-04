"""Foxglove helper messages and visualizer selection."""

from __future__ import annotations

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header

from lidar_interface.pointcloud import from_pointcloud2
from lidar_interface.types import PointCloud
from mapping_visualizer.colors import DYNAMIC, HIGH, LOW
from mapping_visualizer.foxglove import PathRecorder, RelativePose, colored_map, pack_rgb
from mapping_visualizer.launching import visualizer_nodes


def test_pack_rgb_uses_pcl_byte_order() -> None:
    packed = pack_rgb(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    assert packed.tolist() == [0xFF0000, 0x00FF00, 0x0000FF]


def test_colored_map_keeps_points_and_colors_by_confidence() -> None:
    cloud = PointCloud(
        xyz=np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
        stamp_sec=1.0,
        frame_id="map",
        intensity=np.array([0.0, 5.0, 5.0]),
        label=np.array([0.0, 0.0, 1.0]),
        extras={"age": np.array([0.5, 1.0, 2.0])},
    )
    header = Header()
    header.frame_id = "map"
    message = colored_map(cloud, header, 0.4, 1.5)
    assert [field.name for field in message.fields] == ["x", "y", "z", "rgb", "confidence", "dynamic", "age"]
    decoded = from_pointcloud2(message)
    assert np.allclose(decoded.xyz, cloud.xyz)
    assert np.allclose(decoded.extras["age"], [0.5, 1.0, 2.0])
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(3, message.point_step)
    rgb_offset = next(field.offset for field in message.fields if field.name == "rgb")
    packed = raw[:, rgb_offset : rgb_offset + 4].copy().view("<u4").reshape(-1)
    assert packed.tolist() == pack_rgb(np.vstack([LOW, HIGH, DYNAMIC])).tolist()


def test_relative_pose_starts_at_identity() -> None:
    relative = RelativePose()
    start = Rotation.from_euler("z", 90, degrees=True)
    position, rotation = relative(np.array([10.0, 0.0, 0.0]), start)
    assert np.allclose(position, 0.0) and np.allclose(rotation.as_quat(), [0, 0, 0, 1])
    position, _rotation = relative(np.array([10.0, 2.0, 0.0]), start)
    assert np.allclose(position, [2.0, 0.0, 0.0])


def test_path_recorder_spaces_and_caps_poses() -> None:
    recorder = PathRecorder("map", spacing=0.5, max_poses=3)
    kept = []
    for x in [0.0, 0.1, 0.6, 1.2, 1.3, 2.0, 3.0]:
        pose = PoseStamped()
        pose.pose.position.x = x
        kept.append(recorder.add(pose))
    assert kept == [True, False, True, True, False, True, True]
    assert [pose.pose.position.x for pose in recorder.path.poses] == [1.2, 2.0, 3.0]


def test_visualizer_selection() -> None:
    files = ["/workspace/config/visualization.yaml"]
    assert [node.node_executable for node in visualizer_nodes("foxglove", files)] == ["foxglove_bridge", "foxglove_helper"]
    assert [node.node_executable for node in visualizer_nodes("viser", files)] == ["viser_visualizer"]
    assert visualizer_nodes("none", files) == []
    with pytest.raises(ValueError):
        visualizer_nodes("rviz", files)
