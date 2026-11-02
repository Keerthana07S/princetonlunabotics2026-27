# CPU-only ROS 2 Humble image for amd64 laptops and Jetson Orin Nano (JetPack 6 / Ubuntu 22.04).
# The mapping pipeline does not use CUDA. Build this image natively on the target CPU.
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CYCLONEDDS_URI=file:///opt/lunar/cyclonedds.xml \
    ROS_LOCALHOST_ONLY=1 \
    ROS_DOMAIN_ID=42 \
    PYTHONUNBUFFERED=1 \
    LUNAR_CONFIG_DIR=/workspace/config \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libomp-dev \
        lsb-release \
        pkg-config \
        python3-colcon-common-extensions \
        python3-dev \
        python3-opencv \
        python3-pip \
        python3-pytest \
        python3-setuptools \
        python3-wheel \
        ros-humble-cv-bridge \
        ros-humble-diagnostic-msgs \
        ros-humble-launch-ros \
        ros-humble-nav-msgs \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-tf2-geometry-msgs \
        ros-humble-tf2-ros \
        ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /opt/lunar/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /opt/lunar/requirements.txt \
    && python3 -c "import numpy,scipy,open3d,viser,yaml,pytest,cv2,lunarloc; assert numpy.__version__.startswith('1.'), numpy.__version__"

COPY docker/smoke_clipper.py /opt/lunar/smoke_clipper.py
COPY docker/build_clipper.sh /opt/lunar/build_clipper.sh
COPY docker/cyclonedds.xml /opt/lunar/cyclonedds.xml
RUN sed -i 's/\r$//' /opt/lunar/build_clipper.sh \
    && chmod +x /opt/lunar/build_clipper.sh \
    && bash /opt/lunar/build_clipper.sh \
    && rm -rf /var/lib/apt/lists/*
COPY docker/entrypoint.sh /opt/lunar/entrypoint.sh
RUN sed -i 's/\r$//' /opt/lunar/entrypoint.sh && chmod +x /opt/lunar/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/opt/lunar/entrypoint.sh"]
CMD ["bash"]
