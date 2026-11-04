"""Remap a Unitree L2 PointCloud2 topic. This node does not load the vendor SDK."""

from __future__ import annotations

import logging

import rclpy

from lidar_interface.nodes.cloud_relay_node import CloudRelayNode


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = CloudRelayNode(node_name="unitree_adapter", default_input="/unilidar/cloud")
    node.get_logger().info(
        "Unitree adapter is a PointCloud2 remap. The unilidar SDK is not part of this image."
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
