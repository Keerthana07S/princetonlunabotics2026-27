# Networking

The tested layout is one container. Every ROS node in the launch file shares that container's localhost, with `ROS_LOCALHOST_ONLY=1`, `ROS_DOMAIN_ID=42`, and Cyclone DDS (`rmw_cyclonedds_cpp`). The DDS config is `docker/cyclonedds.xml`.

Docker Desktop on Windows does not provide Linux host networking. Multicast between the Windows host and a container is not the supported path. Run `ros2 topic` and `ros2 launch` inside the container:

```bash
docker compose exec dev bash
```

Viser listens on `0.0.0.0:8080` inside the container. Compose publishes that port, so a browser on Windows opens `http://localhost:8080`.

Do not run the `dev` service and the `run` profile at the same time. Both publish port 8080.

The repository lives under OneDrive. The bind mount at `/workspace` works, and it is slower than a local Linux disk.

On the Jetson Orin Nano (JetPack 6, Ubuntu 22.04, aarch64), build the same Dockerfile natively. Do not copy the amd64 image onto the Orin. The Jetson override only raises the memory limit. It does not turn on CUDA. The mapping code is CPU-only.

```bash
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build dev
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d dev
```

JetPack 5 is Ubuntu 20.04 and is outside this image.
