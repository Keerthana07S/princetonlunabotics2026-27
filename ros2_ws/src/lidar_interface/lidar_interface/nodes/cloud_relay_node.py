"""Republish an external PointCloud2 stream through the generic limits."""

from __future__ import annotations

import logging

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from lidar_interface.filters import SensorLimits, apply_limits
from lidar_interface.pointcloud import from_pointcloud2, to_pointcloud2


class CloudRelayNode(Node):
    """Validate an incoming cloud and publish it on the pipeline's input topic."""

    def __init__(self, node_name: str = "cloud_relay", default_input: str = "/points_raw") -> None:
        super().__init__(node_name)
        self.declare_parameter("input_topic", default_input)
        self.declare_parameter("output_topic", "/lidar/points")
        self.declare_parameter("min_range", 0.2)
        self.declare_parameter("max_range", 30.0)
        self.declare_parameter("horizontal_fov_deg", 360.0)
        self.declare_parameter("horizontal_center_deg", 0.0)
        self.declare_parameter("vertical_min_deg", -90.0)
        self.declare_parameter("vertical_max_deg", 90.0)
        output = str(self.get_parameter("output_topic").value)
        input_topic = str(self.get_parameter("input_topic").value)
        self._publisher = self.create_publisher(PointCloud2, output, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, input_topic, self._on_cloud, qos_profile_sensor_data)
        self.get_logger().info("Relaying %s to %s", input_topic, output)

    def _limits(self) -> SensorLimits:
        return SensorLimits(
            min_range=float(self.get_parameter("min_range").value),
            max_range=float(self.get_parameter("max_range").value),
            horizontal_fov_deg=float(self.get_parameter("horizontal_fov_deg").value),
            horizontal_center_deg=float(self.get_parameter("horizontal_center_deg").value),
            vertical_min_deg=float(self.get_parameter("vertical_min_deg").value),
            vertical_max_deg=float(self.get_parameter("vertical_max_deg").value),
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        try:
            cloud = from_pointcloud2(message)
            cloud, stats = apply_limits(cloud, self._limits())
        except Exception as exc:
            self.get_logger().error("Rejected point cloud: %s", exc)
            return
        if cloud.count == 0:
            self.get_logger().warning(
                "Cloud from %s was empty after filtering (input=%d nonfinite=%d)",
                message.header.frame_id,
                stats["input"],
                stats["nonfinite"],
            )
            return
        self._publisher.publish(to_pointcloud2(cloud))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = CloudRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
