import glob
import os

from setuptools import find_packages, setup

package_name = "lidar_interface"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob.glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Princeton Lunabotics",
    maintainer_email="lunabotics@princeton.edu",
    description="Generic PointCloud2 LiDAR interface and synthetic scans.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "synthetic_lidar = lidar_interface.nodes.synthetic_lidar_node:main",
            "cloud_relay = lidar_interface.nodes.cloud_relay_node:main",
            "unitree_adapter = lidar_interface.nodes.unitree_adapter_node:main",
            "stereo_cloud = lidar_interface.nodes.stereo_cloud_node:main",
            "lac_player = lidar_interface.nodes.lac_player_node:main",
            "lac_to_bag = lidar_interface.lac_bag:main",
        ],
    },
)
