"""Publish synthetic LiDAR scans of a moving rover."""

from __future__ import annotations

import logging

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from lidar_interface.filters import SensorLimits, apply_limits
from lidar_interface.geometry import quaternion_xyzw
from lidar_interface.metrics import MetricsLogger
from lidar_interface.pointcloud import stamp_from_seconds, to_pointcloud2
from lidar_interface.synthetic_scene import (
    DEMO_PATH_LENGTH,
    LidarModel,
    RoverPose,
    circle_pose,
    demo_scene,
    line_pose,
)
from lidar_interface.types import PointCloud


class SyntheticLidarNode(Node):
    """Simulate a rover-mounted LiDAR and publish ``sensor_msgs/PointCloud2``."""

    def __init__(self) -> None:
        super().__init__("synthetic_lidar")
        self.declare_parameter("topic", "/lidar/points")
        self.declare_parameter("ground_truth_topic", "/sim/ground_truth_pose")
        self.declare_parameter("frame_id", "lidar_link")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("path", "line")
        self.declare_parameter("speed_m_s", 0.3)
        self.declare_parameter("motion_delay_sec", 0.0)
        self.declare_parameter("path_length_m", DEMO_PATH_LENGTH)
        self.declare_parameter("yaw_rate_rad_s", 0.15)
        self.declare_parameter("min_range", 0.2)
        self.declare_parameter("max_range", 20.0)
        self.declare_parameter("horizontal_fov_deg", 360.0)
        self.declare_parameter("horizontal_center_deg", 0.0)
        self.declare_parameter("vertical_min_deg", -25.0)
        self.declare_parameter("vertical_max_deg", 15.0)
        self.declare_parameter("azimuth_bins", 240)
        self.declare_parameter("elevation_bins", 16)
        self.declare_parameter("range_noise_std", 0.01)
        self.declare_parameter("seed", 7)
        self.declare_parameter("drop_every_n", 0)
        self.declare_parameter("metrics_path", "/workspace/logs/metrics.jsonl")

        topic = str(self.get_parameter("topic").value)
        self._publisher = self.create_publisher(PointCloud2, topic, qos_profile_sensor_data)
        self._truth = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("ground_truth_topic").value),
            10,
        )
        self._scene = demo_scene()
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._index = 0
        self._start = self.get_clock().now()
        self._metrics = MetricsLogger(str(self.get_parameter("metrics_path").value))
        rate = float(self.get_parameter("rate_hz").value)
        if rate <= 0.0:
            raise ValueError("rate_hz must be positive")
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"Synthetic LiDAR publishing on {topic} at {rate:.2f} Hz")

    def _model(self) -> LidarModel:
        return LidarModel(
            min_range=float(self.get_parameter("min_range").value),
            max_range=float(self.get_parameter("max_range").value),
            horizontal_fov_deg=float(self.get_parameter("horizontal_fov_deg").value),
            horizontal_center_deg=float(self.get_parameter("horizontal_center_deg").value),
            vertical_min_deg=float(self.get_parameter("vertical_min_deg").value),
            vertical_max_deg=float(self.get_parameter("vertical_max_deg").value),
            azimuth_bins=int(self.get_parameter("azimuth_bins").value),
            elevation_bins=int(self.get_parameter("elevation_bins").value),
            range_noise_std=float(self.get_parameter("range_noise_std").value),
            seed=int(self.get_parameter("seed").value),
        )

    def _limits(self) -> SensorLimits:
        model = self._model()
        return SensorLimits(
            min_range=model.min_range,
            max_range=model.max_range,
            horizontal_fov_deg=model.horizontal_fov_deg,
            horizontal_center_deg=model.horizontal_center_deg,
            vertical_min_deg=model.vertical_min_deg,
            vertical_max_deg=model.vertical_max_deg,
        )

    def _pose(self, time_sec: float) -> RoverPose:
        path = str(self.get_parameter("path").value)
        speed = float(self.get_parameter("speed_m_s").value)
        if path == "circle":
            return circle_pose(time_sec, speed, float(self.get_parameter("yaw_rate_rad_s").value))
        if path != "line":
            self.get_logger().warning(f"Unknown path '{path}'; using a straight line")
        return line_pose(time_sec, speed, float(self.get_parameter("path_length_m").value))

    def _tick(self) -> None:
        self._index += 1
        drop_every = int(self.get_parameter("drop_every_n").value)
        if drop_every > 0 and self._index % drop_every == 0:
            self.get_logger().warning(f"Dropping synthetic scan {self._index}")
            return
        started = self.get_clock().now()
        elapsed = (started - self._start).nanoseconds * 1e-9
        delay = float(self.get_parameter("motion_delay_sec").value)
        time_sec = max(0.0, elapsed - delay)
        try:
            pose = self._scene.sensor_pose(self._pose(time_sec))
            points = self._scene.scan(pose, time_sec, self._model())
            cloud = PointCloud(xyz=points, stamp_sec=started.nanoseconds * 1e-9, frame_id=self._frame_id)
            cloud, _stats = apply_limits(cloud, self._limits())
        except Exception as exc:
            self.get_logger().error(f"Synthetic scan failed: {exc}")
            return
        if cloud.count == 0:
            self.get_logger().warning(f"Synthetic scan {self._index} produced no valid points")
            return
        self._publisher.publish(to_pointcloud2(cloud))
        self._publish_truth(pose, cloud.stamp_sec)
        elapsed = (self.get_clock().now() - started).nanoseconds * 1e-9
        self._metrics.record(
            "synthetic_scan",
            duration_s=elapsed,
            points=cloud.count,
            scan_index=self._index,
            stamp=cloud.stamp_sec,
            scene_time=time_sec,
            truth_x=float(pose.position[0]),
            truth_y=float(pose.position[1]),
            truth_z=float(pose.position[2]),
            truth_yaw=float(pose.yaw),
        )

    def _publish_truth(self, pose: RoverPose, stamp_sec: float) -> None:
        message = PoseStamped()
        message.header.stamp = stamp_from_seconds(stamp_sec)
        message.header.frame_id = "map"
        message.pose.position.x = float(pose.position[0])
        message.pose.position.y = float(pose.position[1])
        message.pose.position.z = float(pose.position[2])
        quat = quaternion_xyzw(pose.rotation)
        message.pose.orientation.x = float(quat[0])
        message.pose.orientation.y = float(quat[1])
        message.pose.orientation.z = float(quat[2])
        message.pose.orientation.w = float(quat[3])
        self._truth.publish(message)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = SyntheticLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
