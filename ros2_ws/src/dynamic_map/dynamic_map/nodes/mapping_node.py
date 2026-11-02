"""ROS node that registers scans and updates the voxel map."""

from __future__ import annotations

import logging
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

from dynamic_map.session import MappingSession, SessionConfig
from dynamic_map.voxel_map import MapConfig
from lidar_interface.geometry import invert_transform, make_transform, quaternion_xyzw, yaw_rotation
from lidar_interface.metrics import MetricsLogger
from lidar_interface.pointcloud import from_pointcloud2, stamp_from_seconds, to_pointcloud2
from lidar_interface.types import PointCloud
from pointcloud_processing.clipper_association import ClipperConfig
from pointcloud_processing.features import FeatureConfig
from pointcloud_processing.registration import RegistrationConfig


class MappingNode(Node):
    """LiDAR odometry and a dynamic voxel map. This node is not a SLAM system."""

    def __init__(self) -> None:
        super().__init__("mapping")
        self._declare_parameters()
        self._session = MappingSession(self._session_config())
        self._metrics = MetricsLogger(str(self.get_parameter("metrics_path").value))
        self._broadcaster = TransformBroadcaster(self)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE
            if bool(self.get_parameter("map_reliable").value)
            else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._registered = self.create_publisher(
            PointCloud2, str(self.get_parameter("registered_topic").value), qos_profile_sensor_data
        )
        self._global = self.create_publisher(PointCloud2, str(self.get_parameter("map_topic").value), map_qos)
        self._pose = self.create_publisher(Odometry, str(self.get_parameter("pose_topic").value), 10)
        self._status = self.create_publisher(
            DiagnosticStatus, str(self.get_parameter("status_topic").value), 10
        )
        self._lines = self.create_publisher(Marker, str(self.get_parameter("correspondence_topic").value), 10)
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        topic = self.get_parameter("input_topic").value
        mode = self.get_parameter("registration_mode").value
        self.get_logger().info(f"Mapping {topic} in {mode} mode")

    def _declare_parameters(self) -> None:
        defaults = {
            "input_topic": "/mapping/processed_scan",
            "registered_topic": "/mapping/registered_scan",
            "map_topic": "/mapping/global_map",
            "pose_topic": "/mapping/rover_pose",
            "status_topic": "/mapping/registration_status",
            "correspondence_topic": "/mapping/correspondences",
            "map_frame": "map",
            "base_frame": "base_link",
            "map_reliable": True,
            "registration_mode": "scan_to_scan",
            "submap_radius": 8.0,
            "min_points": 30,
            "lidar_x": 0.0,
            "lidar_y": 0.0,
            "lidar_z": 0.0,
            "lidar_yaw": 0.0,
            "keypoint_voxel": 0.35,
            "normal_radius": 0.8,
            "fpfh_radius": 1.2,
            "max_correspondences": 100,
            "max_descriptor_distance": 2.0,
            "mutual_correspondences": False,
            "clipper_sigma": 0.05,
            "clipper_epsilon": 0.25,
            "clipper_mindist": 0.05,
            "clipper_min_inliers": 6,
            "max_rmse": 0.12,
            "min_fitness": 0.5,
            "inlier_threshold": 0.4,
            "icp_enabled": True,
            "icp_distance": 0.5,
            "icp_iterations": 30,
            "voxel_resolution": 0.2,
            "observation_weight": 0.35,
            "decay_lambda": 0.05,
            "max_confidence": 2.0,
            "min_confidence": 0.15,
            "max_voxels": 200000,
            "min_static_observations": 4,
            "motion_threshold": 0.15,
            "free_weight": 0.3,
            "free_hits_to_clear": 2,
            "metrics_path": "/workspace/logs/metrics.jsonl",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _session_config(self) -> SessionConfig:
        def number(name: str) -> float:
            return float(self.get_parameter(name).value)

        return SessionConfig(
            features=FeatureConfig(
                keypoint_voxel=number("keypoint_voxel"),
                normal_radius=number("normal_radius"),
                fpfh_radius=number("fpfh_radius"),
                max_correspondences=int(self.get_parameter("max_correspondences").value),
                max_descriptor_distance=number("max_descriptor_distance"),
                mutual=bool(self.get_parameter("mutual_correspondences").value),
            ),
            clipper=ClipperConfig(
                sigma=number("clipper_sigma"),
                epsilon=number("clipper_epsilon"),
                mindist=number("clipper_mindist"),
                min_inliers=int(self.get_parameter("clipper_min_inliers").value),
            ),
            registration=RegistrationConfig(
                max_rmse=number("max_rmse"),
                min_fitness=number("min_fitness"),
                inlier_threshold=number("inlier_threshold"),
                min_points=int(self.get_parameter("clipper_min_inliers").value),
                icp_enabled=bool(self.get_parameter("icp_enabled").value),
                icp_distance=number("icp_distance"),
                icp_iterations=int(self.get_parameter("icp_iterations").value),
            ),
            voxel=MapConfig(
                resolution=number("voxel_resolution"),
                observation_weight=number("observation_weight"),
                decay_lambda=number("decay_lambda"),
                max_confidence=number("max_confidence"),
                min_confidence=number("min_confidence"),
                max_voxels=int(self.get_parameter("max_voxels").value),
                min_static_observations=int(self.get_parameter("min_static_observations").value),
                motion_threshold=number("motion_threshold"),
                free_weight=number("free_weight"),
                free_hits_to_clear=int(self.get_parameter("free_hits_to_clear").value),
            ),
            mode=str(self.get_parameter("registration_mode").value),
            submap_radius=number("submap_radius"),
            min_points=int(self.get_parameter("min_points").value),
        )

    def _extrinsic(self) -> np.ndarray:
        translation = np.array(
            [
                float(self.get_parameter("lidar_x").value),
                float(self.get_parameter("lidar_y").value),
                float(self.get_parameter("lidar_z").value),
            ]
        )
        return make_transform(yaw_rotation(float(self.get_parameter("lidar_yaw").value)), translation)

    def _on_cloud(self, message: PointCloud2) -> None:
        started = time.perf_counter()
        try:
            cloud = from_pointcloud2(message)
            result = self._session.process_scan(cloud.xyz, cloud.stamp_sec)
        except Exception as exc:
            self.get_logger().error(f"Mapping update failed: {exc}")
            self._publish_status(False, "exception", float("inf"), 0.0, 0, cloud_stamp=self.get_clock().now().nanoseconds * 1e-9)
            return
        self._publish_status(result.accepted, result.reason, result.rmse, result.fitness, result.inliers, cloud.stamp_sec)
        if result.accepted:
            self.get_logger().info(
                f"Accepted scan ({result.reason}) rmse={result.rmse:.4f} inliers={result.inliers} voxels={len(self._session.map.voxels)}"
            )
        if not result.accepted:
            self._metrics.record(
                "registration_rejected",
                duration_s=time.perf_counter() - started,
                **{key: value for key, value in result.timings.items()},
            )
            self.get_logger().warning(f"Kept the previous map ({result.reason})")
            return
        base = result.transform @ invert_transform(self._extrinsic())
        self._publish_pose(base, cloud.stamp_sec)
        self._publish_tf(base, cloud.stamp_sec)
        registered = transform_registered(cloud.xyz, result.transform, cloud.stamp_sec)
        self._registered.publish(to_pointcloud2(registered))
        exported = self._session.map.export(cloud.stamp_sec)
        global_cloud = PointCloud(
            xyz=exported.xyz,
            stamp_sec=cloud.stamp_sec,
            frame_id=str(self.get_parameter("map_frame").value),
            intensity=exported.confidence,
            label=exported.dynamic.astype(np.float64),
        )
        self._global.publish(to_pointcloud2(global_cloud))
        self._lines.publish(self._marker(result.correspondence_lines, cloud.stamp_sec))
        rss = _rss_mb()
        self._metrics.record(
            "mapping",
            duration_s=time.perf_counter() - started,
            features_s=float(result.timings.get("features_s", 0.0)),
            clipper_s=float(result.timings.get("clipper_s", 0.0)),
            registration_s=float(result.timings.get("registration_s", 0.0)),
            map_s=float(result.timings.get("map_s", 0.0)),
            points=cloud.count,
            voxels=len(self._session.map.voxels),
            inliers=result.inliers,
            rmse=result.rmse,
            rss_mb=rss,
        )

    def _publish_pose(self, base: np.ndarray, stamp: float) -> None:
        message = Odometry()
        message.header.stamp = stamp_from_seconds(stamp)
        message.header.frame_id = str(self.get_parameter("map_frame").value)
        message.child_frame_id = str(self.get_parameter("base_frame").value)
        message.pose.pose.position.x = float(base[0, 3])
        message.pose.pose.position.y = float(base[1, 3])
        message.pose.pose.position.z = float(base[2, 3])
        quat = quaternion_xyzw(base)
        message.pose.pose.orientation.x = float(quat[0])
        message.pose.pose.orientation.y = float(quat[1])
        message.pose.pose.orientation.z = float(quat[2])
        message.pose.pose.orientation.w = float(quat[3])
        self._pose.publish(message)

    def _publish_tf(self, base: np.ndarray, stamp: float) -> None:
        transform = TransformStamped()
        transform.header.stamp = stamp_from_seconds(stamp)
        transform.header.frame_id = str(self.get_parameter("map_frame").value)
        transform.child_frame_id = str(self.get_parameter("base_frame").value)
        transform.transform.translation.x = float(base[0, 3])
        transform.transform.translation.y = float(base[1, 3])
        transform.transform.translation.z = float(base[2, 3])
        quat = quaternion_xyzw(base)
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        self._broadcaster.sendTransform(transform)
        extrinsic = self._extrinsic()
        sensor = TransformStamped()
        sensor.header.stamp = transform.header.stamp
        sensor.header.frame_id = str(self.get_parameter("base_frame").value)
        sensor.child_frame_id = "lidar_link"
        sensor.transform.translation.x = float(extrinsic[0, 3])
        sensor.transform.translation.y = float(extrinsic[1, 3])
        sensor.transform.translation.z = float(extrinsic[2, 3])
        sensor_quat = quaternion_xyzw(extrinsic)
        sensor.transform.rotation.x = float(sensor_quat[0])
        sensor.transform.rotation.y = float(sensor_quat[1])
        sensor.transform.rotation.z = float(sensor_quat[2])
        sensor.transform.rotation.w = float(sensor_quat[3])
        self._broadcaster.sendTransform(sensor)

    def _publish_status(
        self,
        accepted: bool,
        reason: str,
        rmse: float,
        fitness: float,
        inliers: int,
        cloud_stamp: float,
    ) -> None:
        status = DiagnosticStatus()
        status.name = "lidar_registration"
        status.hardware_id = "lidar"
        status.level = DiagnosticStatus.OK if accepted else DiagnosticStatus.ERROR
        status.message = reason
        status.values = [
            KeyValue(key="rmse", value=f"{rmse:.5f}"),
            KeyValue(key="fitness", value=f"{fitness:.3f}"),
            KeyValue(key="inliers", value=str(inliers)),
            KeyValue(key="voxels", value=str(len(self._session.map.voxels))),
            KeyValue(key="stamp", value=f"{cloud_stamp:.3f}"),
        ]
        self._status.publish(status)

    def _marker(self, lines: np.ndarray, stamp: float) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp_from_seconds(stamp)
        marker.header.frame_id = str(self.get_parameter("map_frame").value)
        marker.ns = "clipper"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.1
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point

        for segment in lines:
            for endpoint in segment:
                point = Point()
                point.x = float(endpoint[0])
                point.y = float(endpoint[1])
                point.z = float(endpoint[2])
                marker.points.append(point)
        return marker


def transform_registered(points: np.ndarray, transform: np.ndarray, stamp: float) -> PointCloud:
    """Express an accepted scan in the map frame."""
    from lidar_interface.geometry import transform_points

    return PointCloud(
        xyz=transform_points(points, transform),
        stamp_sec=stamp,
        frame_id="map",
    )


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return -1.0
    return -1.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
