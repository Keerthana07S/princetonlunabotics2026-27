import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from lunarloc import FrameDataReader, CameraDataReader
import os
import time


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
        rate = self.create_rate(5)
        for idx, row in self.reader.frames.iterrows():
            ts = self.get_clock().now().to_msg()

            pose_msg = PoseStamped()
            pose_msg.header.stamp = ts
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = float(row['pose.position.x'])
            pose_msg.pose.position.y = float(row['pose.position.y'])
            pose_msg.pose.position.z = float(row['pose.position.z'])
            pose_msg.pose.orientation.x = float(row['pose.orientation.x'])
            pose_msg.pose.orientation.y = float(row['pose.orientation.y'])
            pose_msg.pose.orientation.z = float(row['pose.orientation.z'])
            pose_msg.pose.orientation.w = float(row['pose.orientation.w'])
            self.pose_pub.publish(pose_msg)

            imu_msg = Imu()
            imu_msg.header.stamp = ts
            imu_msg.header.frame_id = 'imu_link'
            imu_msg.linear_acceleration.x = float(row['imu.accel.x'])
            imu_msg.linear_acceleration.y = float(row['imu.accel.y'])
            imu_msg.linear_acceleration.z = float(row['imu.accel.z'])
            imu_msg.angular_velocity.x = float(row['imu.gyro.x'])
            imu_msg.angular_velocity.y = float(row['imu.gyro.y'])
            imu_msg.angular_velocity.z = float(row['imu.gyro.z'])
            self.imu_pub.publish(imu_msg)

            left_img = self.cam_reader.get_image('FrontLeft', idx, 'grayscale')
            right_img = self.cam_reader.get_image('FrontRight', idx, 'grayscale')

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
            rate.sleep()


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
