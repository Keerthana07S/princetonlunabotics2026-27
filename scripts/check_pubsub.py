#!/usr/bin/env python3
"""Two-process DDS check is done by the shell wrapper. This is the in-process check."""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main() -> None:
    rclpy.init()
    node = Node("lunar_pubsub_smoke")
    received: list[str] = []
    node.create_subscription(String, "/lunar/smoke", lambda message: received.append(message.data), 10)
    publisher = node.create_publisher(String, "/lunar/smoke", 10)
    deadline = time.time() + 5.0
    while time.time() < deadline and not received:
        message = String()
        message.data = "phase1"
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if not received or received[-1] != "phase1":
        raise SystemExit("ROS pub/sub smoke failed")
    print("pubsub ok", received[-1])


if __name__ == "__main__":
    main()
