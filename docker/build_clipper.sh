#!/usr/bin/env bash
# Build the official CLIPPER library and its Python module.
# Eigen 3.3.9 is used because Ubuntu 22.04's Eigen 3.4 has an upstream
# double-free report (mit-acl/clipper#24). -mavx/-mfma are removed so the
# same sources compile on the Jetson Orin Nano (aarch64).
set -euo pipefail

CLIPPER_COMMIT="e514dc29c273837ffdfeebbefbdcb2a93d970969"

apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  libomp-dev \
  lsb-release \
  pkg-config \
  libblas-dev \
  liblapack-dev \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel

if [[ ! -f /opt/eigen/share/eigen3/cmake/Eigen3Config.cmake ]]; then
  curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz" \
    -o /tmp/eigen-3.3.9.tar.gz
  rm -rf /opt/eigen-src
  mkdir -p /opt/eigen-src
  tar -xzf /tmp/eigen-3.3.9.tar.gz -C /opt/eigen-src --strip-components=1
  cmake -S /opt/eigen-src -B /tmp/eigen-build -DCMAKE_INSTALL_PREFIX=/opt/eigen
  cmake --install /tmp/eigen-build
  rm -rf /tmp/eigen-build /tmp/eigen-3.3.9.tar.gz
fi

if [[ ! -d /opt/clipper/.git ]]; then
  rm -rf /opt/clipper
  git clone https://github.com/mit-acl/clipper.git /opt/clipper
fi
git -C /opt/clipper fetch --depth 1 origin "${CLIPPER_COMMIT}"
git -C /opt/clipper checkout "${CLIPPER_COMMIT}"

# Upstream forces x86 AVX flags. -march=native covers the CPU that builds the image.
sed -i 's/-march=native -mavx -mfma/-march=native/' /opt/clipper/CMakeLists.txt

cmake -S /opt/clipper -B /opt/clipper/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DEigen3_DIR=/opt/eigen/share/eigen3/cmake \
  -DCLIPPER_BUILD_BINDINGS_PYTHON=ON \
  -DCLIPPER_BUILD_BINDINGS_MATLAB=OFF \
  -DCLIPPER_BUILD_TESTS=ON \
  -DCLIPPER_BUILD_BENCHMARKS=OFF \
  -DCLIPPER_ENABLE_MKL=OFF \
  -DCLIPPER_ENABLE_BLAS=OFF \
  -DCLIPPER_ENABLE_MAXCLIQUE=OFF \
  -DCLIPPER_ENABLE_SCS_SDR=ON

cmake --build /opt/clipper/build --parallel "$(nproc)"
(cd /opt/clipper/build && ctest --output-on-failure)
python3 -m pip install --no-cache-dir /opt/clipper/build/bindings/python
python3 /opt/lunar/smoke_clipper.py

echo "CLIPPER ${CLIPPER_COMMIT} installed"
