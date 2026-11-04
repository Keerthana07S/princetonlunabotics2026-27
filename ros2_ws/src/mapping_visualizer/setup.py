from glob import glob

from setuptools import find_packages, setup

package_name = "mapping_visualizer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Princeton Lunabotics",
    maintainer_email="lunabotics@princeton.edu",
    description="Foxglove and Viser visualization of the live map.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "viser_visualizer = mapping_visualizer.nodes.viser_node:main",
            "foxglove_helper = mapping_visualizer.nodes.foxglove_helper_node:main",
        ],
    },
)
