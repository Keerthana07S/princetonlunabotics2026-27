# Networking

The tested layout is one container. Every ROS node in the launch file shares that container's localhost, with `ROS_LOCALHOST_ONLY=1`, `ROS_DOMAIN_ID=42`, and Cyclone DDS (`rmw_cyclonedds_cpp`). The DDS config is `docker/cyclonedds.xml`.

Docker Desktop on Windows does not provide Linux host networking. Multicast between the Windows host and a container is not the supported path. Run `ros2 topic` and `ros2 launch` inside the container:

```bash
docker compose exec dev bash
```

`foxglove_bridge` listens on `0.0.0.0:8765` inside the container, and the older Viser page (`visualizer:=viser`) on `0.0.0.0:8080`. Compose publishes both ports, so the Foxglove app on Windows connects to `ws://localhost:8765` and a browser opens `http://localhost:8080`. The bridge is a normal ROS node inside the container, so no DDS traffic has to cross into Windows. Version 3.x of the bridge uses the `foxglove.sdk.v1` websocket subprotocol; use a current Foxglove or Lichtblick release.

Do not run the `dev` service and the `run` profile at the same time. Both publish ports 8765 and 8080.

The repository lives under OneDrive. The bind mount at `/workspace` works, and it is slower than a local Linux disk.

On the Jetson Orin Nano (JetPack 6, Ubuntu 22.04, aarch64), build the same Dockerfile natively. Do not copy the amd64 image onto the Orin. The Jetson override only raises the memory limit. It does not turn on CUDA. The mapping code is CPU-only.

```bash
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build dev
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d dev
```

JetPack 5 is Ubuntu 20.04 and is outside this image.
