"""Turn a rectified stereo image pair into ``sensor_msgs/PointCloud2``."""

from __future__ import annotations

import logging
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2

from lidar_interface.metrics import MetricsLogger
from lidar_interface.pointcloud import seconds_from_stamp, to_pointcloud2
from lidar_interface.stereo import StereoConfig, stereo_to_points
from lidar_interface.types import PointCloud


class StereoCloudNode(Node):
    """Pair left/right images by stamp, run SGBM, and publish the left-camera cloud."""

    def __init__(self) -> None:
        super().__init__("stereo_cloud")
        self.declare_parameter("left_topic", "/camera/left/image_raw")
        self.declare_parameter("right_topic", "/camera/right/image_raw")
        self.declare_parameter("output_topic", "/lidar/points")
        self.declare_parameter("frame_id", "lidar_link")
        self.declare_parameter("max_stamp_skew_s", 0.02)
        defaults = StereoConfig()
        self.declare_parameter("horizontal_fov_rad", defaults.horizontal_fov_rad)
        self.declare_parameter("baseline", defaults.baseline)
        self.declare_parameter("scale", defaults.scale)
        self.declare_parameter("num_disparities", defaults.num_disparities)
        self.declare_parameter("block_size", defaults.block_size)
        self.declare_parameter("uniqueness_ratio", defaults.uniqueness_ratio)
        self.declare_parameter("min_depth", defaults.min_depth)
        self.declare_parameter("max_depth", defaults.max_depth)
        self.declare_parameter("min_intensity", defaults.min_intensity)
        self.declare_parameter("stride", defaults.stride)
        self.declare_parameter("metrics_path", "/workspace/logs/metrics_stereo_cloud.jsonl")

        self._bridge = CvBridge()
        self._left: Image | None = None
        self._right: Image | None = None
        self._metrics = MetricsLogger(str(self.get_parameter("metrics_path").value))
        output = str(self.get_parameter("output_topic").value)
        self._publisher = self.create_publisher(PointCloud2, output, qos_profile_sensor_data)
        self.create_subscription(Image, str(self.get_parameter("left_topic").value), self._on_left, 10)
        self.create_subscription(Image, str(self.get_parameter("right_topic").value), self._on_right, 10)
        self.get_logger().info(f"Stereo clouds on {output}")

    def _config(self) -> StereoConfig:
        def value(name: str):
            return self.get_parameter(name).value

        return StereoConfig(
            horizontal_fov_rad=float(value("horizontal_fov_rad")),
            baseline=float(value("baseline")),
            scale=float(value("scale")),
            num_disparities=int(value("num_disparities")),
            block_size=int(value("block_size")),
            uniqueness_ratio=int(value("uniqueness_ratio")),
            min_depth=float(value("min_depth")),
            max_depth=float(value("max_depth")),
            min_intensity=int(value("min_intensity")),
            stride=int(value("stride")),
        )

    def _on_left(self, message: Image) -> None:
        self._left = message
        self._try_pair()

    def _on_right(self, message: Image) -> None:
        self._right = message
        self._try_pair()

    def _try_pair(self) -> None:
        if self._left is None or self._right is None:
            return
        left_stamp = seconds_from_stamp(self._left.header.stamp)
        right_stamp = seconds_from_stamp(self._right.header.stamp)
        if abs(left_stamp - right_stamp) > float(self.get_parameter("max_stamp_skew_s").value):
            return
        left, right = self._left, self._right
        self._left = None
        self._right = None
        started = time.perf_counter()
        try:
            left_image = self._bridge.imgmsg_to_cv2(left, desired_encoding="mono8")
            right_image = self._bridge.imgmsg_to_cv2(right, desired_encoding="mono8")
            points = stereo_to_points(left_image, right_image, self._config())
        except Exception as exc:
            self.get_logger().error(f"Stereo matching failed: {exc}")
            return
        cloud = PointCloud(xyz=points, stamp_sec=left_stamp, frame_id=str(self.get_parameter("frame_id").value))
        self._publisher.publish(to_pointcloud2(cloud))
        self._metrics.record(
            "stereo_cloud",
            duration_s=time.perf_counter() - started,
            points=cloud.count,
            stamp=left_stamp,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = StereoCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
