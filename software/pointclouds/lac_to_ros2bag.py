import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from lunarloc import FrameDataReader, CameraDataReader
import math
import os
import time


def _number(row, *names):
    for name in names:
        if name in row.index:
            return float(row[name])
    raise KeyError(names[0])


def _orientation(row):
    """Return x, y, z, w. Euler columns are roll, pitch, yaw in radians."""
    if "pose.orientation.w" in row.index:
        return (
            _number(row, "pose.orientation.x"),
            _number(row, "pose.orientation.y"),
            _number(row, "pose.orientation.z"),
            _number(row, "pose.orientation.w"),
        )
    roll = _number(row, "roll")
    pitch = _number(row, "pitch")
    yaw = _number(row, "yaw")
    half_roll, half_pitch, half_yaw = roll * 0.5, pitch * 0.5, yaw * 0.5
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class LACPublisher(Node):
    def __init__(self, lac_path):
        super().__init__('lac_publisher')
        self.bridge = CvBridge()

        self.pose_pub = self.create_publisher(PoseStamped, '/ground_truth/pose', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.left_img_pub = self.create_publisher(Image, '/camera/left/image_raw', 10)
        self.right_img_pub = self.create_publisher(Image, '/camera/right/image_raw', 10)

        self.reader = FrameDataReader(lac_path)
        self.cam_reader = CameraDataReader(lac_path)

        self.publish_all_frames()

    def publish_all_frames(self):
        # time.sleep, not create_rate().sleep(): the rate timer never fires
        # until an executor is spinning, and this loop runs before rclpy.spin.
        for idx, row in self.reader.frames.iterrows():
            ts = self.get_clock().now().to_msg()

            pose_msg = PoseStamped()
            pose_msg.header.stamp = ts
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = _number(row, "pose.position.x", "x")
            pose_msg.pose.position.y = _number(row, "pose.position.y", "y")
            pose_msg.pose.position.z = _number(row, "pose.position.z", "z")
            quat_x, quat_y, quat_z, quat_w = _orientation(row)
            pose_msg.pose.orientation.x = quat_x
            pose_msg.pose.orientation.y = quat_y
            pose_msg.pose.orientation.z = quat_z
            pose_msg.pose.orientation.w = quat_w
            self.pose_pub.publish(pose_msg)

            imu_msg = Imu()
            imu_msg.header.stamp = ts
            imu_msg.header.frame_id = 'imu_link'
            imu_msg.linear_acceleration.x = _number(row, "imu.accel.x", "accel_x")
            imu_msg.linear_acceleration.y = _number(row, "imu.accel.y", "accel_y")
            imu_msg.linear_acceleration.z = _number(row, "imu.accel.z", "accel_z")
            imu_msg.angular_velocity.x = _number(row, "imu.gyro.x", "gyro_x")
            imu_msg.angular_velocity.y = _number(row, "imu.gyro.y", "gyro_y")
            imu_msg.angular_velocity.z = _number(row, "imu.gyro.z", "gyro_z")
            self.imu_pub.publish(imu_msg)

            frame = int(row["frame"]) if "frame" in row.index else int(idx)
            left_img = self.cam_reader.get_image('FrontLeft', frame, 'grayscale')
            right_img = self.cam_reader.get_image('FrontRight', frame, 'grayscale')

            if left_img is not None:
                left_msg = self.bridge.cv2_to_imgmsg(left_img, encoding='mono8')
                left_msg.header.stamp = ts
                left_msg.header.frame_id = 'camera_left'
                self.left_img_pub.publish(left_msg)

            if right_img is not None:
                right_msg = self.bridge.cv2_to_imgmsg(right_img, encoding='mono8')
                right_msg.header.stamp = ts
                right_msg.header.frame_id = 'camera_right'
                self.right_img_pub.publish(right_msg)

            self.get_logger().info(f'Published frame {idx}')
            time.sleep(0.2)


def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: ros2 run your_package lac_publisher.py <path_to_lac>')
        return

    rclpy.init()
    node = LACPublisher(sys.argv[1])
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
