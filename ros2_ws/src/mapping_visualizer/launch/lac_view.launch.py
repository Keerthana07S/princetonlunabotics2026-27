"""View a LAC recording: a .lac file directly, or a bag made by ``lac_to_bag``.

    ros2 launch mapping_visualizer lac_view.launch.py
    ros2 launch mapping_visualizer lac_view.launch.py log:="/path/to/run.lac" rate:=2.0 loop:=true
    ros2 launch mapping_visualizer lac_view.launch.py log:=/path/to/run_bag stereo:=false
    ros2 launch mapping_visualizer lac_view.launch.py visualizer:=viser

With Foxglove (the default), connect to ws://localhost:8765 and open
config/foxglove/lunar_mapping.json. It shows the left camera image, the
ground-truth path, and the current stereo cloud on the rover. There is no
mapping; use ``dynamic_map stereo_pipeline.launch.py`` for that. With
``stereo:=false`` no stereo matching runs, which is what you want for a bag
written with ``--stereo`` (its clouds are already on ``/lidar/points``).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node

from mapping_visualizer.launching import DEFAULT_VISUALIZER, visualizer_nodes

DEFAULT_LOG = "/workspace/software/pointclouds/example (2).lac"


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _nodes(context):
    config = os.environ.get("LUNAR_CONFIG_DIR", "/workspace/config")
    stereo_config = os.path.join(config, "stereo")
    log = context.launch_configurations["log"]
    rate = float(context.launch_configurations["rate"])
    loop = _truthy(context.launch_configurations["loop"])
    stereo = _truthy(context.launch_configurations["stereo"])

    actions = []
    if log.endswith(".lac"):
        if not os.path.isfile(log):
            raise FileNotFoundError(log)
        wait_for = ["/camera/left/image_raw", "/ground_truth/pose"] + (["/camera/right/image_raw"] if stereo else [])
        actions.append(
            Node(
                package="lidar_interface",
                executable="lac_player",
                name="lac_player",
                output="screen",
                parameters=[
                    os.path.join(stereo_config, "lac.yaml"),
                    {"lac_path": log, "playback_rate": rate, "loop": loop, "wait_for_topics": wait_for, "metrics_path": ""},
                ],
            )
        )
    else:
        if not os.path.isdir(log):
            raise FileNotFoundError(f"{log} is neither a .lac file nor a bag directory")
        command = ["ros2", "bag", "play", log, "--rate", str(rate), "--delay", "5.0"]
        actions.append(ExecuteProcess(cmd=command + (["--loop"] if loop else []), output="screen"))

    if stereo:
        actions.append(
            Node(
                package="lidar_interface",
                executable="stereo_cloud",
                name="stereo_cloud",
                output="screen",
                parameters=[os.path.join(stereo_config, "lac.yaml"), {"metrics_path": ""}],
            )
        )
    actions += visualizer_nodes(
        context.launch_configurations["visualizer"],
        [os.path.join(config, "visualization.yaml"), os.path.join(stereo_config, "visualization.yaml")],
        viser_overrides={"live_cloud_topic": "/lidar/points"},
        helper_overrides={"publish_ground_truth_tf": True},
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("log", default_value=DEFAULT_LOG, description=".lac file or rosbag2 directory"),
            DeclareLaunchArgument("rate", default_value="1.0", description="playback speed"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("stereo", default_value="true", description="run stereo matching on the images"),
            DeclareLaunchArgument("visualizer", default_value=DEFAULT_VISUALIZER, description="foxglove, viser, or none"),
            OpaqueFunction(function=_nodes),
        ]
    )
