"""Launch the stereo mapping pipeline on a LAC log, with a visualizer.

lac_player -> /camera/{left,right}/image_raw -> stereo_cloud -> /lidar/points
-> preprocessing -> /mapping/processed_scan -> mapping -> Foxglove or Viser.

Every stage after stereo_cloud is the LiDAR pipeline. Stereo values come from
``config/stereo/*.yaml``, loaded after the shared files so they override them.
Set ``use_lac_player:=false`` to feed live camera topics instead, and
``visualizer:=viser`` for the Viser page instead of foxglove_bridge.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mapping_visualizer.launching import DEFAULT_VISUALIZER, visualizer_nodes


def _visualizer(context, config: str):
    files = [os.path.join(config, "visualization.yaml"), os.path.join(config, "stereo", "visualization.yaml")]
    return visualizer_nodes(context.launch_configurations["visualizer"], files)


def generate_launch_description() -> LaunchDescription:
    config = os.environ.get("LUNAR_CONFIG_DIR", "/workspace/config")
    stereo = os.path.join(config, "stereo")
    player_overrides = {
        "lac_path": LaunchConfiguration("lac_path"),
        "playback_rate": LaunchConfiguration("playback_rate"),
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument("lac_path", default_value="/workspace/software/pointclouds/example (2).lac"),
            DeclareLaunchArgument("playback_rate", default_value="1.0"),
            DeclareLaunchArgument("use_lac_player", default_value="true"),
            DeclareLaunchArgument("visualizer", default_value=DEFAULT_VISUALIZER, description="foxglove, viser, or none"),
            Node(
                package="lidar_interface",
                executable="lac_player",
                name="lac_player",
                output="screen",
                parameters=[os.path.join(stereo, "lac.yaml"), player_overrides],
                condition=IfCondition(LaunchConfiguration("use_lac_player")),
            ),
            Node(
                package="lidar_interface",
                executable="stereo_cloud",
                name="stereo_cloud",
                output="screen",
                parameters=[os.path.join(stereo, "lac.yaml")],
            ),
            Node(
                package="pointcloud_processing",
                executable="preprocessing",
                name="preprocessing",
                output="screen",
                parameters=[os.path.join(config, "preprocessing.yaml"), os.path.join(stereo, "preprocessing.yaml")],
            ),
            Node(
                package="dynamic_map",
                executable="mapping",
                name="mapping",
                output="screen",
                parameters=[
                    os.path.join(config, "features.yaml"),
                    os.path.join(config, "clipper.yaml"),
                    os.path.join(config, "registration.yaml"),
                    os.path.join(config, "mapping.yaml"),
                    os.path.join(stereo, "mapping.yaml"),
                ],
            ),
            OpaqueFunction(function=_visualizer, args=[config]),
        ]
    )
