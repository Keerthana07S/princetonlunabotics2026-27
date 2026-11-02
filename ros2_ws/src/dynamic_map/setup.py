import glob
import os

from setuptools import find_packages, setup

package_name = "dynamic_map"

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
    description="Dynamic voxel map and LiDAR odometry.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mapping = dynamic_map.nodes.mapping_node:main",
        ],
    },
)
