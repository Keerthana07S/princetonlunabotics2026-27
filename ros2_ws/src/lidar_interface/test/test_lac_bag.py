"""LAC message builders, and the bag converter against the example recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lidar_interface.lac import LacFrame
from lidar_interface.lac_messages import GROUND_TRUTH_TOPIC, LEFT_TOPIC, mono8_message, pose_message

LOG = Path(__file__).resolve().parents[4] / "software" / "pointclouds" / "example (2).lac"


def test_mono8_message_keeps_pixels_and_rejects_color() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    message = mono8_message(image, 12.5, "camera_left")
    assert (message.height, message.width, message.step, message.encoding) == (3, 4, 4, "mono8")
    assert bytes(message.data) == image.tobytes()
    assert message.header.stamp.sec == 12 and message.header.stamp.nanosec == 500_000_000
    with pytest.raises(ValueError):
        mono8_message(np.zeros((3, 4, 3), dtype=np.uint8), 0.0, "camera_left")


def test_pose_message_carries_rover_transform() -> None:
    rover = np.eye(4)
    rover[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    rover[:3, 3] = [1.0, 2.0, 3.0]
    frame = LacFrame(frame=1, mission_time=0.0, rover=rover, accel=np.zeros(3), gyro=np.zeros(3), has_stereo=False)
    message = pose_message(frame, 0.0)
    position = message.pose.position
    orientation = message.pose.orientation
    assert (position.x, position.y, position.z) == (1.0, 2.0, 3.0)
    assert np.allclose([orientation.z, orientation.w], [np.sqrt(0.5), np.sqrt(0.5)])


@pytest.mark.skipif(not LOG.exists(), reason="example LAC log is not present")
def test_convert_writes_a_readable_bag(tmp_path: Path) -> None:
    pytest.importorskip("lunarloc")
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from lidar_interface.lac_bag import convert

    output = tmp_path / "bag"
    counts = convert(str(LOG), str(output), start_time=100.0)
    assert counts[LEFT_TOPIC] == 60

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    read: dict[str, int] = {}
    stamps = []
    while reader.has_next():
        topic, _data, stamp = reader.read_next()
        read[topic] = read.get(topic, 0) + 1
        stamps.append(stamp)
    assert read == counts
    assert read[GROUND_TRUTH_TOPIC] > read[LEFT_TOPIC]
    assert stamps[0] == 100_000_000_000
    assert all(later >= earlier for earlier, later in zip(stamps, stamps[1:]))
