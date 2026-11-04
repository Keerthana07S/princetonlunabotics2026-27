"""Launch the synthetic mapping pipeline and a visualizer.

``visualizer:=foxglove`` (default) starts foxglove_bridge on ws://localhost:8765;
``visualizer:=viser`` starts the Viser page on http://localhost:8080.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

from mapping_visualizer.launching import DEFAULT_VISUALIZER, visualizer_nodes


def _visualizer(context, config: str):
    return visualizer_nodes(
        context.launch_configurations["visualizer"], [os.path.join(config, "visualization.yaml")]
    )


def generate_launch_description() -> LaunchDescription:
    config = os.environ.get("LUNAR_CONFIG_DIR", "/workspace/config")
    return LaunchDescription(
        [
            DeclareLaunchArgument("visualizer", default_value=DEFAULT_VISUALIZER, description="foxglove, viser, or none"),
            Node(
                package="lidar_interface",
                executable="synthetic_lidar",
                name="synthetic_lidar",
                output="screen",
                parameters=[os.path.join(config, "lidar.yaml")],
            ),
            Node(
                package="pointcloud_processing",
                executable="preprocessing",
                name="preprocessing",
                output="screen",
                parameters=[os.path.join(config, "preprocessing.yaml")],
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
                ],
            ),
            OpaqueFunction(function=_visualizer, args=[config]),
        ]
    )
