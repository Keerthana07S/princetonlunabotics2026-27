"""Convert a Lunar Autonomy Challenge (.lac) log into a ROS 2 bag.

The bag holds the same topics as the ``lac_player`` node, so ``ros2 bag play``
can stand in for it:

- ``/camera/left/image_raw`` and ``/camera/right/image_raw`` (``sensor_msgs/Image``, mono8)
- ``/ground_truth/pose`` (``geometry_msgs/PoseStamped``, rover pose in the LAC world frame)
- ``/imu/data`` (``sensor_msgs/Imu``)
- ``/lidar/points`` (``sensor_msgs/PointCloud2``), only with ``--stereo``

Every stamp is ``--start-time`` plus the row's mission time, and the bag's
record time equals the header stamp.

    ros2 run lidar_interface lac_to_bag "example (2).lac" example_bag --stereo
    ros2 bag info example_bag
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import yaml

from lidar_interface.lac import LacLog
from lidar_interface.lac_messages import (
    GROUND_TRUTH_TOPIC,
    IMU_TOPIC,
    LEFT_TOPIC,
    RIGHT_TOPIC,
    imu_message,
    mono8_message,
    pose_message,
)

STEREO_TOPIC = "/lidar/points"
TOPIC_TYPES = {
    LEFT_TOPIC: "sensor_msgs/msg/Image",
    RIGHT_TOPIC: "sensor_msgs/msg/Image",
    GROUND_TRUTH_TOPIC: "geometry_msgs/msg/PoseStamped",
    IMU_TOPIC: "sensor_msgs/msg/Imu",
    STEREO_TOPIC: "sensor_msgs/msg/PointCloud2",
}


def stereo_config(path: Path | None):
    """``StereoConfig`` from the ``stereo_cloud`` section of a parameter file, or the defaults."""
    from lidar_interface.stereo import StereoConfig

    if path is None or not path.exists():
        return StereoConfig()
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = (content.get("stereo_cloud") or {}).get("ros__parameters") or {}
    fields = StereoConfig.__dataclass_fields__
    return StereoConfig(**{key: value for key, value in values.items() if key in fields})


def convert(
    lac_path: str,
    output: str,
    stereo: bool = False,
    stereo_config_path: Path | None = None,
    all_rows: bool = False,
    start_time: float | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write the bag and return the message count per topic."""
    import rosbag2_py
    from rclpy.serialization import serialize_message

    target = Path(output)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite to replace it")
        shutil.rmtree(target)
    log = LacLog(lac_path)
    frames = log.frames
    if not all_rows:
        first = next((index for index, frame in enumerate(frames) if frame.has_stereo), 0)
        frames = frames[first:]
    if not frames:
        raise ValueError(f"{lac_path} has no rows")
    origin = frames[0].mission_time
    start = time.time() if start_time is None else float(start_time)
    config = stereo_config(stereo_config_path) if stereo else None

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(target), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = [LEFT_TOPIC, RIGHT_TOPIC, GROUND_TRUTH_TOPIC, IMU_TOPIC] + ([STEREO_TOPIC] if stereo else [])
    for topic in topics:
        writer.create_topic(
            rosbag2_py.TopicMetadata(name=topic, type=TOPIC_TYPES[topic], serialization_format="cdr")
        )
    counts = {topic: 0 for topic in topics}

    def write(topic: str, message, stamp_sec: float) -> None:
        writer.write(topic, serialize_message(message), int(round(stamp_sec * 1e9)))
        counts[topic] += 1

    for frame in frames:
        stamp = start + frame.mission_time - origin
        write(GROUND_TRUTH_TOPIC, pose_message(frame, stamp), stamp)
        write(IMU_TOPIC, imu_message(frame, stamp), stamp)
        if not frame.has_stereo:
            continue
        pair = log.stereo_pair(frame.frame)
        if pair is None:
            continue
        write(LEFT_TOPIC, mono8_message(pair[0], stamp, "camera_left"), stamp)
        write(RIGHT_TOPIC, mono8_message(pair[1], stamp, "camera_right"), stamp)
        if config is not None:
            from lidar_interface.pointcloud import to_pointcloud2
            from lidar_interface.stereo import stereo_to_points
            from lidar_interface.types import PointCloud

            points = stereo_to_points(pair[0], pair[1], config)
            write(STEREO_TOPIC, to_pointcloud2(PointCloud(xyz=points, stamp_sec=stamp, frame_id="lidar_link")), stamp)
    del writer
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lac", help="path to the .lac file")
    parser.add_argument("output", nargs="?", help="bag directory to create (default: <lac name>_bag next to the log)")
    parser.add_argument("--stereo", action="store_true", help="also write stereo clouds on /lidar/points")
    parser.add_argument(
        "--stereo-config",
        type=Path,
        default=Path(os.environ.get("LUNAR_CONFIG_DIR", "/workspace/config")) / "stereo" / "lac.yaml",
        help="parameter file whose stereo_cloud section sets the stereo matcher",
    )
    parser.add_argument("--all-rows", action="store_true", help="keep rows before the first stereo pair")
    parser.add_argument("--start-time", type=float, help="Unix time of the first row (default: now)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    args = parser.parse_args(argv)

    lac = Path(args.lac)
    output = args.output or str(lac.with_name(lac.stem.replace(" ", "_").replace("(", "").replace(")", "") + "_bag"))
    started = time.perf_counter()
    counts = convert(
        str(lac),
        output,
        stereo=args.stereo,
        stereo_config_path=args.stereo_config,
        all_rows=args.all_rows,
        start_time=args.start_time,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output} in {time.perf_counter() - started:.1f} s")
    for topic, count in counts.items():
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
