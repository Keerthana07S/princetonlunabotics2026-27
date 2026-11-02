"""ROS node that filters ``/lidar/points`` and publishes ``/mapping/processed_scan``."""

from __future__ import annotations

import logging
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from lidar_interface.metrics import MetricsLogger
from lidar_interface.pointcloud import from_pointcloud2, to_pointcloud2
from lidar_interface.types import PointCloud
from pointcloud_processing.preprocessing import PreprocessConfig, preprocess


class PreprocessingNode(Node):
    """Preserve the incoming stamp and frame id."""

    def __init__(self) -> None:
        super().__init__("preprocessing")
        self.declare_parameter("input_topic", "/lidar/points")
        self.declare_parameter("output_topic", "/mapping/processed_scan")
        self.declare_parameter("min_range", 0.2)
        self.declare_parameter("max_range", 20.0)
        self.declare_parameter("voxel_size", 0.15)
        self.declare_parameter("statistical_enabled", True)
        self.declare_parameter("statistical_neighbors", 16)
        self.declare_parameter("statistical_std_ratio", 2.0)
        self.declare_parameter("radius_enabled", False)
        self.declare_parameter("radius", 0.4)
        self.declare_parameter("radius_min_neighbors", 4)
        self.declare_parameter("segment_enabled", False)
        self.declare_parameter("cluster_eps", 0.35)
        self.declare_parameter("cluster_min_points", 8)
        self.declare_parameter("min_cluster_size", 30)
        self.declare_parameter("plane_removal_enabled", False)
        self.declare_parameter("plane_distance", 0.08)
        self.declare_parameter("metrics_path", "/workspace/logs/metrics.jsonl")
        self._metrics = MetricsLogger(str(self.get_parameter("metrics_path").value))
        output = str(self.get_parameter("output_topic").value)
        input_topic = str(self.get_parameter("input_topic").value)
        self._publisher = self.create_publisher(PointCloud2, output, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, input_topic, self._on_cloud, qos_profile_sensor_data)
        self.get_logger().info(f"Preprocessing {input_topic} -> {output}")

    def _config(self) -> PreprocessConfig:
        return PreprocessConfig(
            min_range=float(self.get_parameter("min_range").value),
            max_range=float(self.get_parameter("max_range").value),
            voxel_size=float(self.get_parameter("voxel_size").value),
            statistical_enabled=bool(self.get_parameter("statistical_enabled").value),
            statistical_neighbors=int(self.get_parameter("statistical_neighbors").value),
            statistical_std_ratio=float(self.get_parameter("statistical_std_ratio").value),
            radius_enabled=bool(self.get_parameter("radius_enabled").value),
            radius=float(self.get_parameter("radius").value),
            radius_min_neighbors=int(self.get_parameter("radius_min_neighbors").value),
            segment_enabled=bool(self.get_parameter("segment_enabled").value),
            cluster_eps=float(self.get_parameter("cluster_eps").value),
            cluster_min_points=int(self.get_parameter("cluster_min_points").value),
            min_cluster_size=int(self.get_parameter("min_cluster_size").value),
            plane_removal_enabled=bool(self.get_parameter("plane_removal_enabled").value),
            plane_distance=float(self.get_parameter("plane_distance").value),
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        started = time.perf_counter()
        try:
            cloud = from_pointcloud2(message)
            filtered = preprocess(cloud.xyz, self._config())
        except Exception as exc:
            self.get_logger().error(f"Preprocessing failed: {exc}")
            return
        output = PointCloud(xyz=filtered, stamp_sec=cloud.stamp_sec, frame_id=cloud.frame_id)
        self._publisher.publish(to_pointcloud2(output))
        self._metrics.record(
            "preprocessing",
            duration_s=time.perf_counter() - started,
            input_points=cloud.count,
            output_points=output.count,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = PreprocessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
