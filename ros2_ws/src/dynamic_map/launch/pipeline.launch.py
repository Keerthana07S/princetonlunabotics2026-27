"""Launch the synthetic mapping pipeline and the Viser process."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = os.environ.get("LUNAR_CONFIG_DIR", "/workspace/config")
    return LaunchDescription(
        [
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
            Node(
                package="mapping_visualizer",
                executable="viser_visualizer",
                name="mapping_visualizer",
                output="screen",
                parameters=[os.path.join(config, "visualization.yaml")],
            ),
        ]
    )
