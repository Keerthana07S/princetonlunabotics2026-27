"""Replay a boulder traverse CSV through CLIPPER and draw every candidate pair.

Each detection frame publishes the candidate pairs between detections (placed
with the predicted pose, before correction) and map landmarks, split by
CLIPPER's decision and by ground-truth correctness:

- ``kept_correct``: CLIPPER kept it and both ends are the same boulder
- ``kept_wrong``: CLIPPER kept it but the ends are different boulders
- ``dropped_correct``: a true match that CLIPPER left out
- ``dropped_wrong``: a false candidate that CLIPPER left out

Open config/foxglove/boulder_clipper.json as the layout. Namespaces can be
toggled under each topic in the 3D panel settings.
"""

from __future__ import annotations

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Float64
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from pointcloud_processing.boulder_association import BoulderConfig, BoulderSession, FrameResult, load_traverse
from pointcloud_processing.clipper_association import ClipperConfig

PAIR_STYLES = {
    "kept_correct": ((0.15, 0.9, 0.3, 1.0), 0.04),
    "kept_wrong": ((1.0, 0.1, 0.1, 1.0), 0.05),
    "dropped_correct": ((1.0, 0.6, 0.0, 0.9), 0.025),
    "dropped_wrong": ((0.6, 0.6, 0.6, 0.45), 0.01),
}


def _color(rgba: tuple[float, float, float, float]) -> ColorRGBA:
    return ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))


def _point(xyz: np.ndarray) -> Point:
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _quaternion_z(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))


class BoulderClipperNode(Node):
    def __init__(self) -> None:
        super().__init__("boulder_clipper")
        defaults = BoulderConfig()
        self.declare_parameter("csv_path", "/workspace/traverse_1 (1).csv")
        self.declare_parameter("frames_per_second", 5.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("wait_for_subscriber", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("truth_frame", "truth_link")
        self.declare_parameter("label_landmarks", True)
        self.declare_parameter("seed", defaults.seed)
        self.declare_parameter("scale_error", defaults.scale_error)
        self.declare_parameter("translation_noise", defaults.translation_noise)
        self.declare_parameter("yaw_drift_deg_per_m", defaults.yaw_drift_deg_per_m)
        self.declare_parameter("yaw_noise_deg_per_m", defaults.yaw_noise_deg_per_m)
        self.declare_parameter("identity_radius", defaults.identity_radius)
        self.declare_parameter("gate_radius", defaults.gate_radius)
        self.declare_parameter("max_candidates_per_detection", defaults.max_candidates_per_detection)
        self.declare_parameter("max_rmse", defaults.max_rmse)
        self.declare_parameter("max_correction_yaw_deg", defaults.max_correction_yaw_deg)
        self.declare_parameter("max_correction_m", defaults.max_correction_m)
        self.declare_parameter("merge_radius", defaults.merge_radius)
        self.declare_parameter("correct", defaults.correct)
        self.declare_parameter("clipper_sigma", defaults.clipper.sigma)
        self.declare_parameter("clipper_epsilon", defaults.clipper.epsilon)
        self.declare_parameter("clipper_mindist", defaults.clipper.mindist)
        self.declare_parameter("clipper_min_inliers", defaults.clipper.min_inliers)

        value = lambda name: self.get_parameter(name).value  # noqa: E731
        config = BoulderConfig(
            seed=int(value("seed")),
            scale_error=float(value("scale_error")),
            translation_noise=float(value("translation_noise")),
            yaw_drift_deg_per_m=float(value("yaw_drift_deg_per_m")),
            yaw_noise_deg_per_m=float(value("yaw_noise_deg_per_m")),
            identity_radius=float(value("identity_radius")),
            gate_radius=float(value("gate_radius")),
            max_candidates_per_detection=int(value("max_candidates_per_detection")),
            max_rmse=float(value("max_rmse")),
            max_correction_yaw_deg=float(value("max_correction_yaw_deg")),
            max_correction_m=float(value("max_correction_m")),
            merge_radius=float(value("merge_radius")),
            correct=bool(value("correct")),
            clipper=ClipperConfig(
                sigma=float(value("clipper_sigma")),
                epsilon=float(value("clipper_epsilon")),
                mindist=float(value("clipper_mindist")),
                min_inliers=int(value("clipper_min_inliers")),
            ),
        )
        self._map_frame = str(value("map_frame"))
        self._base_frame = str(value("base_frame"))
        self._truth_frame = str(value("truth_frame"))
        self._labels = bool(value("label_landmarks"))
        self._loop = bool(value("loop"))
        self._start_frame = int(value("start_frame"))
        self._waiting = bool(value("wait_for_subscriber"))

        path = str(value("csv_path"))
        self._session = BoulderSession(load_traverse(path), config)
        self.get_logger().info(
            f"{path}: {len(self._session.detection_rows)} detection frames, "
            f"{self._session.identity_centroids.shape[0]} ground-truth boulders"
        )

        self._associations = self.create_publisher(MarkerArray, "/boulders/associations", 10)
        self._detections = self.create_publisher(MarkerArray, "/boulders/detections", 10)
        self._landmarks = self.create_publisher(MarkerArray, "/boulders/landmarks", 10)
        self._paths = {
            name: self.create_publisher(Path, f"/boulders/{name}_path", 1) for name in ("truth", "odometry", "estimated")
        }
        self._status = self.create_publisher(DiagnosticStatus, "/boulders/status", 10)
        self._error = self.create_publisher(Float64, "/boulders/translation_error", 10)
        self._odometry_error = self.create_publisher(Float64, "/boulders/odometry_translation_error", 10)
        self._broadcaster = TransformBroadcaster(self)
        self._reset_paths()
        self._finished = False
        self._last_wait_log = 0.0
        self.create_timer(1.0 / max(float(value("frames_per_second")), 0.1), self._tick)

    def _reset_paths(self) -> None:
        self._path_messages = {name: Path() for name in self._paths}
        for message in self._path_messages.values():
            message.header.frame_id = self._map_frame

    def _tick(self) -> None:
        if self._finished:
            return
        if self._waiting:
            if self._associations.get_subscription_count() == 0:
                now = self.get_clock().now().nanoseconds * 1e-9
                if now - self._last_wait_log > 5.0:
                    self.get_logger().info("Waiting for a subscriber on /boulders/associations (connect Foxglove)")
                    self._last_wait_log = now
                return
            self._waiting = False
        result = self._session.step()
        while result is not None and result.frame < self._start_frame:
            self._append_paths(result)
            result = self._session.step()
        if result is None:
            summary = self._session.summary()
            self.get_logger().info(
                "Traverse done: accepted {accepted}/{detection_frames}, CLIPPER inlier precision "
                "{clipper_inlier_precision}, final error {final_translation_error_m} m "
                "(odometry {odometry_final_translation_error_m} m)".format(**summary)
            )
            if self._loop:
                self._session.reset()
                self._reset_paths()
            else:
                self._finished = True
            return
        self._publish(result)

    def _append_paths(self, result: FrameResult) -> None:
        stamp = self.get_clock().now().to_msg()
        for name, position, yaw in (
            ("truth", result.truth, 0.0),
            ("odometry", result.odometry_position, result.odometry_yaw),
            ("estimated", result.position, result.yaw),
        ):
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._map_frame
            pose.pose.position = _point(position)
            q = _quaternion_z(yaw)
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
            self._path_messages[name].poses.append(pose)

    def _marker(self, namespace: str, kind: int, stamp, scale: float, rgba) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self._map_frame
        marker.ns = namespace
        marker.id = 0
        marker.type = kind
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
        marker.color = _color(rgba)
        return marker

    def _clear(self, stamp) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self._map_frame
        marker.action = Marker.DELETEALL
        return marker

    def _publish(self, result: FrameResult) -> None:
        stamp = self.get_clock().now().to_msg()
        record = result.record

        associations = MarkerArray(markers=[self._clear(stamp)])
        groups = {
            "kept_correct": result.kept & result.correct,
            "kept_wrong": result.kept & ~result.correct,
            "dropped_correct": ~result.kept & result.correct,
            "dropped_wrong": ~result.kept & ~result.correct,
        }
        for namespace, mask in groups.items():
            rgba, width = PAIR_STYLES[namespace]
            marker = self._marker(namespace, Marker.LINE_LIST, stamp, width, rgba)
            for source, target in result.candidates[mask]:
                marker.points.append(_point(result.detections_predicted[source]))
                marker.points.append(_point(result.landmarks[target]))
            associations.markers.append(marker)
        self._associations.publish(associations)

        detections = MarkerArray(markers=[self._clear(stamp)])
        for namespace, points, scale, rgba in (
            ("predicted", result.detections_predicted, 0.12, (0.2, 0.8, 1.0, 1.0)),
            ("corrected", result.detections_corrected, 0.08, (1.0, 1.0, 1.0, 0.9)),
            ("logged", self._session.traverse.boulders[result.row], 0.06, (1.0, 0.2, 0.9, 0.9)),
        ):
            marker = self._marker(namespace, Marker.SPHERE_LIST, stamp, scale, rgba)
            marker.points = [_point(p) for p in points]
            detections.markers.append(marker)
        self._detections.publish(detections)

        landmarks = MarkerArray(markers=[self._clear(stamp)])
        marker = self._marker("landmarks", Marker.SPHERE_LIST, stamp, 0.15, (1.0, 0.85, 0.1, 0.8))
        marker.points = [_point(p) for p in result.landmarks]
        landmarks.markers.append(marker)
        truth = self._marker("ground_truth_boulders", Marker.SPHERE_LIST, stamp, 0.1, (0.6, 0.4, 1.0, 0.35))
        truth.points = [_point(p) for p in self._session.identity_centroids]
        landmarks.markers.append(truth)
        if self._labels:
            for index, (position, identity) in enumerate(zip(result.landmarks, result.landmark_identities)):
                label = self._marker("labels", Marker.TEXT_VIEW_FACING, stamp, 0.12, (1.0, 1.0, 1.0, 0.9))
                label.id = index
                label.pose.position = _point(position + np.array([0.0, 0.0, 0.2]))
                label.text = str(int(identity))
                landmarks.markers.append(label)
        self._landmarks.publish(landmarks)

        self._append_paths(result)
        for name, publisher in self._paths.items():
            self._path_messages[name].header.stamp = stamp
            publisher.publish(self._path_messages[name])

        transforms = []
        for child, position, yaw in (
            (self._base_frame, result.position, result.yaw),
            (self._truth_frame, result.truth, 0.0),
        ):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self._map_frame
            transform.child_frame_id = child
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = (
                float(v) for v in position
            )
            q = _quaternion_z(yaw)
            rotation = transform.transform.rotation
            rotation.x, rotation.y, rotation.z, rotation.w = q
            transforms.append(transform)
        self._broadcaster.sendTransform(transforms)

        status = DiagnosticStatus()
        status.name = "boulder_clipper"
        status.hardware_id = "traverse"
        status.level = DiagnosticStatus.OK if result.accepted else DiagnosticStatus.WARN
        status.message = result.reason
        keys = (
            "frame",
            "detections",
            "candidates",
            "correct_candidates",
            "clipper_inliers",
            "clipper_inliers_correct",
            "fit_rmse",
            "correction_yaw_deg",
            "correction_m",
            "landmarks",
            "translation_error",
            "odometry_translation_error",
            "yaw_deg",
            "odometry_yaw_deg",
        )
        status.values = [
            KeyValue(key=key, value=f"{record[key]:.4f}" if isinstance(record[key], float) else str(record[key]))
            for key in keys
        ]
        groups_count = {name: int(mask.sum()) for name, mask in groups.items()}
        status.values += [KeyValue(key=name, value=str(count)) for name, count in groups_count.items()]
        self._status.publish(status)
        self._error.publish(Float64(data=float(record["translation_error"])))
        self._odometry_error.publish(Float64(data=float(record["odometry_translation_error"])))


def main() -> None:
    rclpy.init()
    node = BoulderClipperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
