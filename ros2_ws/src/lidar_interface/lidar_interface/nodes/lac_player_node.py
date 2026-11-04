"""Replay a Lunar Autonomy Challenge (.lac) log as ROS topics.

Rows are released on a timer at ``playback_rate`` times mission time. Every
row publishes the rover's ground-truth pose and IMU sample; rows with a front
stereo pair also publish both images with the same stamp.

A row's stamp is the wall-clock start of playback plus its mission time
(divided by ``playback_rate``), not the moment it was released. Reading an
image from the archive can stall the timer, and stamping on release would then
squeeze several captures into a fraction of a second, which corrupts the
mapper's constant-velocity prediction.
"""

from __future__ import annotations

import logging
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu

from lidar_interface.lac import LacFrame, LacLog
from lidar_interface.lac_messages import imu_message, mono8_message, pose_message
from lidar_interface.metrics import MetricsLogger


class LacPlayerNode(Node):
    """Publish ``/camera/{left,right}/image_raw``, ``/ground_truth/pose``, and ``/imu/data``."""

    def __init__(self) -> None:
        super().__init__("lac_player")
        self.declare_parameter("lac_path", "")
        self.declare_parameter("playback_rate", 1.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("start_delay_sec", 3.0)
        # Playback waits until every listed topic has a subscriber.
        self.declare_parameter("wait_for_topics", ["/camera/left/image_raw", "/camera/right/image_raw"])
        self.declare_parameter("start_at_first_image", True)
        self.declare_parameter("left_topic", "/camera/left/image_raw")
        self.declare_parameter("right_topic", "/camera/right/image_raw")
        self.declare_parameter("ground_truth_topic", "/ground_truth/pose")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("metrics_path", "/workspace/logs/metrics_lac_player.jsonl")

        path = str(self.get_parameter("lac_path").value)
        if not path:
            raise ValueError("lac_path is required")
        self._rate = float(self.get_parameter("playback_rate").value)
        if self._rate <= 0.0:
            raise ValueError("playback_rate must be positive")
        started = time.perf_counter()
        self._log = LacLog(path)
        frames = self._log.frames
        if bool(self.get_parameter("start_at_first_image").value):
            first = next((index for index, frame in enumerate(frames) if frame.has_stereo), 0)
            frames = frames[first:]
        if not frames:
            raise ValueError(f"{path} has no rows")
        self._frames = frames
        self._next = 0
        self._origin_time = frames[0].mission_time
        self._released_at: float | None = None
        self._stamp_origin = 0.0
        first_pair = next((frame for frame in self._frames if frame.has_stereo), None)
        if first_pair is not None:
            # The first archive read indexes the whole tar; do it before playback.
            self._log.stereo_pair(first_pair.frame)
        self._ready_at = time.monotonic() + float(self.get_parameter("start_delay_sec").value)
        self._metrics = MetricsLogger(str(self.get_parameter("metrics_path").value))

        self._left = self.create_publisher(Image, str(self.get_parameter("left_topic").value), 10)
        self._right = self.create_publisher(Image, str(self.get_parameter("right_topic").value), 10)
        self._truth = self.create_publisher(PoseStamped, str(self.get_parameter("ground_truth_topic").value), 10)
        self._imu = self.create_publisher(Imu, str(self.get_parameter("imu_topic").value), 10)
        self._timer = self.create_timer(0.02, self._tick)
        pairs = sum(frame.has_stereo for frame in self._frames)
        self.get_logger().info(
            f"Loaded {path}: {len(self._frames)} rows, {pairs} stereo pairs, "
            f"{self._frames[-1].mission_time - self._origin_time:.1f} s of mission time "
            f"in {time.perf_counter() - started:.1f} s; playback x{self._rate:g}"
        )

    def _tick(self) -> None:
        if self._released_at is None:
            if time.monotonic() < self._ready_at or not self._subscribers_ready():
                return
            self._start()
        target = self._origin_time + (time.monotonic() - self._released_at) * self._rate
        while self._next < len(self._frames) and self._frames[self._next].mission_time <= target:
            self._publish(self._frames[self._next])
            self._next += 1
        if self._next < len(self._frames):
            return
        if bool(self.get_parameter("loop").value):
            self.get_logger().info("Playback finished; looping")
            self._next = 0
            self._start()
            return
        self.get_logger().info("Playback finished")
        self._timer.cancel()

    def _start(self) -> None:
        self._released_at = time.monotonic()
        self._stamp_origin = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info("Starting playback")

    def _subscribers_ready(self) -> bool:
        topics = [str(topic) for topic in self.get_parameter("wait_for_topics").value if str(topic)]
        return all(self.count_subscribers(topic) > 0 for topic in topics)

    def _publish(self, frame: LacFrame) -> None:
        pair = None
        loaded = time.perf_counter()
        if frame.has_stereo:
            pair = self._log.stereo_pair(frame.frame)
            if pair is None:
                self.get_logger().warning(f"Frame {frame.frame} lists a stereo pair that could not be read")
        load_s = time.perf_counter() - loaded
        stamp = self._stamp_origin + (frame.mission_time - self._origin_time) / self._rate
        self._truth.publish(pose_message(frame, stamp, str(self.get_parameter("map_frame").value)))
        self._imu.publish(imu_message(frame, stamp))
        if pair is None:
            return
        self._left.publish(mono8_message(pair[0], stamp, "camera_left"))
        self._right.publish(mono8_message(pair[1], stamp, "camera_right"))
        roll, pitch, yaw = _euler_xyz(frame.rover)
        self._metrics.record(
            "lac_frame",
            duration_s=load_s,
            frame=frame.frame,
            mission_time=frame.mission_time,
            stamp=stamp,
            truth_x=float(frame.rover[0, 3]),
            truth_y=float(frame.rover[1, 3]),
            truth_z=float(frame.rover[2, 3]),
            truth_roll=roll,
            truth_pitch=pitch,
            truth_yaw=yaw,
        )


def _euler_xyz(transform) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    return float(roll), float(pitch), float(yaw)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rclpy.init()
    node = LacPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
