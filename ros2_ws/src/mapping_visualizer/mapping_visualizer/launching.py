"""Visualizer nodes for launch files, chosen by a ``visualizer`` launch argument."""

from __future__ import annotations

from launch_ros.actions import Node

VISUALIZERS = ("foxglove", "viser", "none")
DEFAULT_VISUALIZER = "foxglove"


def visualizer_nodes(
    kind: str,
    parameter_files: list[str],
    viser_overrides: dict | None = None,
    helper_overrides: dict | None = None,
) -> list[Node]:
    """Nodes for ``kind``. ``parameter_files`` hold the ``mapping_visualizer``, ``foxglove_bridge``, and ``foxglove_helper`` sections."""
    if kind == "none":
        return []
    if kind == "viser":
        return [
            Node(
                package="mapping_visualizer",
                executable="viser_visualizer",
                name="mapping_visualizer",
                output="screen",
                parameters=[*parameter_files, viser_overrides or {}],
            )
        ]
    if kind == "foxglove":
        return [
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=parameter_files,
            ),
            Node(
                package="mapping_visualizer",
                executable="foxglove_helper",
                name="foxglove_helper",
                output="screen",
                parameters=[*parameter_files, helper_overrides or {}],
            ),
        ]
    raise ValueError(f"visualizer must be one of {', '.join(VISUALIZERS)}, not {kind!r}")
