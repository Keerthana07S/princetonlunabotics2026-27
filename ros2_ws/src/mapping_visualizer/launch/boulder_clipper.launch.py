"""Replay a boulder traverse CSV through CLIPPER and view the associations in Foxglove.

    ros2 launch mapping_visualizer boulder_clipper.launch.py
    ros2 launch mapping_visualizer boulder_clipper.launch.py csv:=/workspace/traverse_7.csv rate:=10.0
    ros2 launch mapping_visualizer boulder_clipper.launch.py start_frame:=3000 seed:=1 loop:=true

Connect Foxglove to ws://localhost:8765 and open
config/foxglove/boulder_clipper.json. Playback waits until Foxglove subscribes
to /boulders/associations, so the first frame is not missed.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

DEFAULT_CSV = "/workspace/traverse_1 (1).csv"


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _nodes(context):
    config = os.environ.get("LUNAR_CONFIG_DIR", "/workspace/config")
    values = context.launch_configurations
    csv = values["csv"]
    if not os.path.isfile(csv):
        raise FileNotFoundError(csv)
    parameters = {
        "csv_path": csv,
        "frames_per_second": float(values["rate"]),
        "loop": _truthy(values["loop"]),
        "start_frame": int(values["start_frame"]),
        "seed": int(values["seed"]),
        "correct": _truthy(values["correct"]),
        "wait_for_subscriber": _truthy(values["wait_for_subscriber"]),
    }
    return [
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[os.path.join(config, "visualization.yaml")],
        ),
        Node(
            package="mapping_visualizer",
            executable="boulder_clipper",
            name="boulder_clipper",
            output="screen",
            parameters=[parameters],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("csv", default_value=DEFAULT_CSV, description="traverse CSV"),
            DeclareLaunchArgument("rate", default_value="5.0", description="detection frames per second"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("start_frame", default_value="0", description="skip to this CSV frame without drawing"),
            DeclareLaunchArgument("seed", default_value="0", description="odometry noise seed"),
            DeclareLaunchArgument("correct", default_value="true", description="false runs odometry only"),
            DeclareLaunchArgument("wait_for_subscriber", default_value="true"),
            OpaqueFunction(function=_nodes),
        ]
    )
