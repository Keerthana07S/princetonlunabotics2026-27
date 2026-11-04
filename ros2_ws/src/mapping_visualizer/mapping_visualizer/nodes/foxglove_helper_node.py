"""Topics for the Foxglove view: colored map, trails, and optional ground-truth TF.

Foxglove draws ROS topics through ``foxglove_bridge``. The raw topics already
cover clouds, images, and markers; this node adds what the Viser process used
to compute itself.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from lidar_interface.pointcloud import from_pointcloud2
from mapping_visualizer.foxglove import PathRecorder, RelativePose, colored_map

MAP_TOPIC = "/viz/map"
GROUND_TRUTH_PATH_TOPIC = "/viz/ground_truth_path"
GROUND_TRUTH_POSE_TOPIC = "/viz/ground_truth_pose"
ESTIMATED_PATH_TOPIC = "/viz/estimated_path"


class FoxgloveHelperNode(Node):
    def __init__(self) -> None:
        super().__init__("foxglove_helper")
        self.declare_parameter("map_topic", "/mapping/global_map")
        self.declare_parameter("map_reliable", True)
        self.declare_parameter("low_confidence", 0.4)
        self.declare_parameter("high_confidence", 1.5)
        self.declare_parameter("pose_topic", "/mapping/rover_pose")
        self.declare_parameter("ground_truth_topic", "/ground_truth/pose")
        # Express ground truth relative to its first pose, which is where the map frame starts.
        self.declare_parameter("ground_truth_relative", True)
        self.declare_parameter("map_frame", "map")
        # With no mapping node, broadcast map -> base_link from ground truth and a
        # static base_link -> sensor_frame at sensor_mount, so sensor clouds have a pose.
        self.declare_parameter("publish_ground_truth_tf", False)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("sensor_frame", "lidar_link")
        self.declare_parameter("sensor_mount", [0.0, 0.0, 0.0])
        self.declare_parameter("path_spacing", 0.05)
        self.declare_parameter("max_path_poses", 20000)

        frame = str(self.get_parameter("map_frame").value)
        spacing = float(self.get_parameter("path_spacing").value)
        max_poses = int(self.get_parameter("max_path_poses").value)
        self._truth_path = PathRecorder(frame, spacing, max_poses)
        self._estimate_path = PathRecorder(frame, spacing, max_poses)
        self._relative = RelativePose() if bool(self.get_parameter("ground_truth_relative").value) else None
        self._low = float(self.get_parameter("low_confidence").value)
        self._high = float(self.get_parameter("high_confidence").value)

        reliable = bool(self.get_parameter("map_reliable").value)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._map = self.create_publisher(PointCloud2, MAP_TOPIC, map_qos)
        self._truth_path_pub = self.create_publisher(Path, GROUND_TRUTH_PATH_TOPIC, 1)
        self._truth_pose_pub = self.create_publisher(PoseStamped, GROUND_TRUTH_POSE_TOPIC, 10)
        self._estimate_path_pub = self.create_publisher(Path, ESTIMATED_PATH_TOPIC, 1)
        self.create_subscription(PointCloud2, str(self.get_parameter("map_topic").value), self._on_map, map_qos)
        self.create_subscription(Odometry, str(self.get_parameter("pose_topic").value), self._on_pose, 10)
        self.create_subscription(
            PoseStamped, str(self.get_parameter("ground_truth_topic").value), self._on_ground_truth, 10
        )

        self._broadcaster = None
        if bool(self.get_parameter("publish_ground_truth_tf").value):
            self._broadcaster = TransformBroadcaster(self)
            self._static = StaticTransformBroadcaster(self)
            mount = np.asarray(self.get_parameter("sensor_mount").value, dtype=np.float64).reshape(3)
            sensor = TransformStamped()
            sensor.header.stamp = self.get_clock().now().to_msg()
            sensor.header.frame_id = str(self.get_parameter("base_frame").value)
            sensor.child_frame_id = str(self.get_parameter("sensor_frame").value)
            sensor.transform.translation.x, sensor.transform.translation.y, sensor.transform.translation.z = (
                float(v) for v in mount
            )
            sensor.transform.rotation.w = 1.0
            self._static.sendTransform(sensor)
        self.get_logger().info("Foxglove topics on /viz/*")

    def _on_map(self, message: PointCloud2) -> None:
        try:
            cloud = from_pointcloud2(message)
        except Exception as exc:
            self.get_logger().error(f"Map cloud rejected: {exc}")
            return
        self._map.publish(colored_map(cloud, message.header, self._low, self._high))

    def _on_pose(self, message: Odometry) -> None:
        pose = PoseStamped()
        pose.header = message.header
        pose.pose = message.pose.pose
        if self._estimate_path.add(pose):
            self._estimate_path_pub.publish(self._estimate_path.path)

    def _on_ground_truth(self, message: PoseStamped) -> None:
        pose = message
        if self._relative is not None:
            position = np.array([message.pose.position.x, message.pose.position.y, message.pose.position.z])
            quat = message.pose.orientation
            position, rotation = self._relative(position, Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]))
            pose = PoseStamped()
            pose.header.stamp = message.header.stamp
            pose.header.frame_id = str(self.get_parameter("map_frame").value)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in position)
            x, y, z, w = rotation.as_quat()
            pose.pose.orientation.x, pose.pose.orientation.y = float(x), float(y)
            pose.pose.orientation.z, pose.pose.orientation.w = float(z), float(w)
        self._truth_pose_pub.publish(pose)
        if self._truth_path.add(pose):
            self._truth_path_pub.publish(self._truth_path.path)
        if self._broadcaster is not None:
            transform = TransformStamped()
            transform.header = pose.header
            transform.child_frame_id = str(self.get_parameter("base_frame").value)
            transform.transform.translation.x = pose.pose.position.x
            transform.transform.translation.y = pose.pose.position.y
            transform.transform.translation.z = pose.pose.position.z
            transform.transform.rotation = pose.pose.orientation
            self._broadcaster.sendTransform(transform)


def main() -> None:
    rclpy.init()
    node = FoxgloveHelperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
