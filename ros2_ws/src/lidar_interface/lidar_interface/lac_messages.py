"""ROS messages for one LAC log row. Shared by the live player and the bag writer."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, Imu

from lidar_interface.geometry import quaternion_xyzw
from lidar_interface.lac import LacFrame
from lidar_interface.pointcloud import stamp_from_seconds

LEFT_TOPIC = "/camera/left/image_raw"
RIGHT_TOPIC = "/camera/right/image_raw"
GROUND_TRUTH_TOPIC = "/ground_truth/pose"
IMU_TOPIC = "/imu/data"


def pose_message(frame: LacFrame, stamp_sec: float, frame_id: str = "map") -> PoseStamped:
    """Rover pose in the LAC world frame."""
    message = PoseStamped()
    message.header.stamp = stamp_from_seconds(stamp_sec)
    message.header.frame_id = frame_id
    message.pose.position.x, message.pose.position.y, message.pose.position.z = (float(v) for v in frame.rover[:3, 3])
    quat = quaternion_xyzw(frame.rover)
    message.pose.orientation.x = float(quat[0])
    message.pose.orientation.y = float(quat[1])
    message.pose.orientation.z = float(quat[2])
    message.pose.orientation.w = float(quat[3])
    return message


def imu_message(frame: LacFrame, stamp_sec: float, frame_id: str = "imu_link") -> Imu:
    message = Imu()
    message.header.stamp = stamp_from_seconds(stamp_sec)
    message.header.frame_id = frame_id
    message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = (
        float(v) for v in frame.accel
    )
    message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = (float(v) for v in frame.gyro)
    return message


def mono8_message(image: np.ndarray, stamp_sec: float, frame_id: str) -> Image:
    """``sensor_msgs/Image`` with ``mono8`` encoding from an ``(H, W)`` uint8 array."""
    pixels = np.ascontiguousarray(image, dtype=np.uint8)
    if pixels.ndim != 2:
        raise ValueError("mono8 images must be two-dimensional")
    message = Image()
    message.header.stamp = stamp_from_seconds(stamp_sec)
    message.header.frame_id = frame_id
    message.height, message.width = (int(v) for v in pixels.shape)
    message.encoding = "mono8"
    message.is_bigendian = 0
    message.step = message.width
    message.data = pixels.tobytes()
    return message
