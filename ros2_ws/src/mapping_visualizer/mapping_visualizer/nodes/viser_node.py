"""Viser process. Callbacks store the latest messages; a timer redraws the scene."""

from __future__ import annotations

import logging
import threading

import numpy as np
import rclpy
import viser
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker

from lidar_interface.pointcloud import from_pointcloud2
from mapping_visualizer.colors import SCAN, confidence_colors


class ViserNode(Node):
    """Live 3D view on port 8080. It does not participate in registration."""

    def __init__(self) -> None:
        super().__init__("mapping_visualizer")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("update_hz", 8.0)
        self.declare_parameter("map_topic", "/mapping/global_map")
        self.declare_parameter("scan_topic", "/mapping/registered_scan")
        self.declare_parameter("pose_topic", "/mapping/rover_pose")
        self.declare_parameter("correspondence_topic", "/mapping/correspondences")
        self.declare_parameter("map_reliable", True)
        self.declare_parameter("point_size", 0.06)
        self.declare_parameter("max_points", 40000)
        self.declare_parameter("low_confidence", 0.4)
        self.declare_parameter("high_confidence", 1.5)
        self.declare_parameter("ground_truth_topic", "/ground_truth/pose")
        self.declare_parameter("image_topic", "/camera/left/image_raw")

        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        self._server = viser.ViserServer(host=host, port=port)
        self._map_handle = None
        self._scan_handle = None
        self._line_handle = None
        self._rover = self._server.scene.add_frame("/rover", axes_length=0.45, axes_radius=0.015)
        self._lidar = self._server.scene.add_frame("/lidar", axes_length=0.25, axes_radius=0.01)
        self._status = self._server.gui.add_markdown("Waiting for the map")
        self._image_panel = self._server.gui.add_image(
            np.zeros((8, 8, 3), dtype=np.uint8),
            label="Left camera",
        )
        self._lock = threading.Lock()
        self._map = None
        self._scan = None
        self._pose = None
        self._lines = np.zeros((0, 2, 3), dtype=np.float64)
        self._trajectory = np.zeros((0, 3), dtype=np.float64)
        self._trajectory_handle = None
        self._camera_image = None
        self._camera_count = 0
        self._aimed_clients: set[int] = set()

        reliable = bool(self.get_parameter("map_reliable").value)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(PointCloud2, str(self.get_parameter("map_topic").value), self._on_map, map_qos)
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("scan_topic").value),
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(Odometry, str(self.get_parameter("pose_topic").value), self._on_pose, 10)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("ground_truth_topic").value),
            self._on_ground_truth,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._on_image,
            10,
        )
        self.create_subscription(
            Marker,
            str(self.get_parameter("correspondence_topic").value),
            self._on_lines,
            10,
        )
        rate = float(self.get_parameter("update_hz").value)
        if rate <= 0.0:
            raise ValueError("update_hz must be positive")
        self.create_timer(1.0 / rate, self._draw)
        self.get_logger().info(f"Viser at http://{host}:{port}")

    def _on_map(self, message: PointCloud2) -> None:
        try:
            cloud = from_pointcloud2(message)
        except Exception as exc:
            self.get_logger().error(f"Map cloud rejected: {exc}")
            return
        with self._lock:
            self._map = cloud

    def _on_scan(self, message: PointCloud2) -> None:
        try:
            cloud = from_pointcloud2(message)
        except Exception as exc:
            self.get_logger().error(f"Scan cloud rejected: {exc}")
            return
        with self._lock:
            self._scan = cloud

    def _on_pose(self, message: Odometry) -> None:
        with self._lock:
            self._pose = message

    def _on_ground_truth(self, message: PoseStamped) -> None:
        position = np.array(
            [message.pose.position.x, message.pose.position.y, message.pose.position.z],
            dtype=np.float64,
        )
        quat = message.pose.orientation
        wxyz = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
        with self._lock:
            self._pose = (position, wxyz)
            self._trajectory = np.vstack([self._trajectory, position.reshape(1, 3)])

    def _on_image(self, message: Image) -> None:
        try:
            image = _rgb_image(message)
        except ValueError as exc:
            self.get_logger().error(f"Camera image rejected: {exc}")
            return
        with self._lock:
            self._camera_image = image
            self._camera_count += 1

    def _on_lines(self, message: Marker) -> None:
        if len(message.points) < 2:
            return
        coords = np.array([[point.x, point.y, point.z] for point in message.points], dtype=np.float64)
        with self._lock:
            self._lines = coords.reshape(-1, 2, 3)

    def _draw(self) -> None:
        with self._lock:
            cloud = self._map
            scan = self._scan
            pose = self._pose
            lines = None if self._lines.shape[0] == 0 else self._lines.copy()
            trajectory = self._trajectory.copy()
            camera = None if self._camera_image is None else self._camera_image.copy()
            camera_count = self._camera_count
        point_size = float(self.get_parameter("point_size").value)
        cap = int(self.get_parameter("max_points").value)
        voxels = 0
        if cloud is not None:
            points, confidence, dynamic = _subsample(cloud.xyz, cloud.intensity, cloud.label, cap)
            colors = confidence_colors(
                confidence,
                dynamic,
                float(self.get_parameter("low_confidence").value),
                float(self.get_parameter("high_confidence").value),
            )
            self._map_handle = _upsert_cloud(self._server, self._map_handle, "/map", points, colors, point_size)
            voxels = int(cloud.count)
        if scan is not None:
            points, _confidence, _dynamic = _subsample(scan.xyz, None, None, cap)
            colors = np.repeat(SCAN.reshape(1, 3), points.shape[0], axis=0)
            self._scan_handle = _upsert_cloud(
                self._server, self._scan_handle, "/scan", points, colors, point_size * 1.4
            )
        if pose is not None:
            position, wxyz = _pose_arrays(pose)
            self._rover.position = position
            self._rover.wxyz = wxyz
            self._lidar.position = position
            self._lidar.wxyz = wxyz
        if trajectory.shape[0] > 0:
            trail_colors = np.tile(np.array([[0.15, 0.45, 0.95]]), (trajectory.shape[0], 1))
            self._trajectory_handle = _upsert_cloud(
                self._server, self._trajectory_handle, "/trajectory", trajectory, trail_colors, 0.35
            )
            center = trajectory.mean(axis=0)
            for client in self._server.get_clients().values():
                if client.client_id in self._aimed_clients:
                    continue
                client.camera.position = center + np.array([12.0, -12.0, 8.0])
                client.camera.look_at = center
                self._aimed_clients.add(client.client_id)
        if camera is not None:
            self._image_panel.image = camera
        if lines is not None and lines.shape[0] > 0:
            segments = np.asarray(lines, dtype=np.float64).reshape(-1, 2, 3)
            colors = np.tile(np.array([[[1.0, 0.8, 0.1], [1.0, 0.8, 0.1]]]), (segments.shape[0], 1, 1))
            try:
                self._line_handle = _upsert_lines(
                    self._server, self._line_handle, "/correspondences", segments, colors
                )
            except Exception as exc:
                self.get_logger().error(f"Correspondence display failed: {exc}")
        self._status.content = (
            f"Map voxels: {voxels}. Trajectory points: {trajectory.shape[0]}. "
            f"Left images: {camera_count}."
        )


def _subsample(points, intensity, label, cap: int):
    count = int(points.shape[0])
    if count == 0:
        return np.zeros((1, 3)), np.zeros(1), np.zeros(1, dtype=bool)
    if count > cap:
        index = np.linspace(0, count - 1, cap).astype(int)
    else:
        index = np.arange(count)
    chosen = points[index]
    confidence = np.ones(chosen.shape[0]) if intensity is None else intensity[index]
    dynamic = np.zeros(chosen.shape[0], dtype=bool) if label is None else label[index] > 0.5
    return chosen, confidence, dynamic


def _pose_arrays(pose: Odometry | tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pose, tuple):
        return pose
    position = np.array(
        [pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z],
        dtype=np.float64,
    )
    quat = pose.pose.pose.orientation
    wxyz = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
    return position, wxyz


def _rgb_image(message: Image) -> np.ndarray:
    """Return an ``(H, W, 3)`` uint8 image. Mono frames are repeated into RGB."""
    if message.height == 0 or message.width == 0:
        raise ValueError("empty camera image")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if message.encoding in ("mono8", "8UC1"):
        gray = raw.reshape(message.height, message.width)
        return np.repeat(gray[:, :, None], 3, axis=2)
    if message.encoding == "rgb8":
        return raw.reshape(message.height, message.width, 3).copy()
    if message.encoding == "bgr8":
        bgr = raw.reshape(message.height, message.width, 3)
        return bgr[:, :, ::-1].copy()
    raise ValueError(f"unsupported image encoding {message.encoding}")


def _upsert_cloud(server, handle, name: str, points, colors, size: float):
    if handle is None:
        return server.scene.add_point_cloud(name, points=points, colors=colors, point_size=size)
    handle.points = points
    handle.colors = colors
    handle.point_size = size
    return handle


def _upsert_lines(server, handle, name: str, points, colors):
    if not hasattr(server.scene, "add_line_segments"):
        return handle
    if handle is None:
        return server.scene.add_line_segments(name, points=points, colors=colors, line_width=2.0)
    handle.points = points
    handle.colors = colors
    return handle


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = ViserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
