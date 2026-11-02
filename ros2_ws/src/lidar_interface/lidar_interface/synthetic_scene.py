"""Ray-cast synthetic scenes. This is not a vendor LiDAR model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lidar_interface.geometry import make_transform, yaw_rotation


@dataclass
class Sphere:
    """A sphere that can translate at a constant velocity and appear for a time window."""

    name: str
    center: np.ndarray
    radius: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active_start: float = 0.0
    active_end: float = float("inf")

    def center_at(self, time_sec: float) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64) + np.asarray(self.velocity, dtype=np.float64) * time_sec

    def active(self, time_sec: float) -> bool:
        return self.active_start <= time_sec <= self.active_end


@dataclass
class Box:
    """An axis-aligned box in the world frame."""

    name: str
    minimum: np.ndarray
    maximum: np.ndarray
    active_start: float = 0.0
    active_end: float = float("inf")

    def active(self, time_sec: float) -> bool:
        return self.active_start <= time_sec <= self.active_end


@dataclass
class LidarModel:
    """Generic spinning sample pattern. Angles are degrees and ranges are meters.

    These defaults are a stand-in, not the Unitree L2 pattern. Tune them when a
    real sensor is connected.
    """

    min_range: float = 0.2
    max_range: float = 20.0
    horizontal_fov_deg: float = 360.0
    horizontal_center_deg: float = 0.0
    vertical_min_deg: float = -5.0
    vertical_max_deg: float = 30.0
    azimuth_bins: int = 180
    elevation_bins: int = 16
    range_noise_std: float = 0.01
    seed: int = 7


@dataclass
class RoverPose:
    """Rover pose in the world frame. The LiDAR frame matches the rover frame."""

    position: np.ndarray
    yaw: float

    @property
    def rotation(self) -> np.ndarray:
        return yaw_rotation(self.yaw)

    @property
    def transform(self) -> np.ndarray:
        return make_transform(self.rotation, self.position)


def line_pose(time_sec: float, speed: float) -> RoverPose:
    """Drive forward along +x at a constant speed."""
    return RoverPose(position=np.array([speed * time_sec, 0.0, 0.0]), yaw=0.0)


def circle_pose(time_sec: float, speed: float, yaw_rate: float) -> RoverPose:
    """Drive a circle in the xy plane. ``yaw_rate`` is radians per second."""
    if abs(yaw_rate) < 1e-6:
        return line_pose(time_sec, speed)
    radius = speed / yaw_rate
    yaw = yaw_rate * time_sec
    position = np.array([radius * np.sin(yaw), radius * (1.0 - np.cos(yaw)), 0.0])
    return RoverPose(position=position, yaw=yaw)


def _ray_sphere(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    offset = origin - center
    half_b = direction @ offset
    discriminant = half_b * half_b - (offset @ offset - radius * radius)
    times = np.full(direction.shape[0], np.inf, dtype=np.float64)
    valid = discriminant >= 0.0
    if not np.any(valid):
        return times
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = -half_b - root
    far = -half_b + root
    chosen = np.where(near > 1e-4, near, np.where(far > 1e-4, far, np.inf))
    times[valid] = chosen[valid]
    return times


def _ray_box(origin: np.ndarray, direction: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    count = direction.shape[0]
    t_near = np.full(count, -np.inf, dtype=np.float64)
    t_far = np.full(count, np.inf, dtype=np.float64)
    missed = np.zeros(count, dtype=bool)
    for axis in range(3):
        component = direction[:, axis]
        parallel = np.abs(component) < 1e-12
        outside = parallel & ((origin[axis] < minimum[axis]) | (origin[axis] > maximum[axis]))
        missed |= outside
        inverse = np.zeros(count, dtype=np.float64)
        inverse[~parallel] = 1.0 / component[~parallel]
        first = (minimum[axis] - origin[axis]) * inverse
        second = (maximum[axis] - origin[axis]) * inverse
        axis_near = np.minimum(first, second)
        axis_far = np.maximum(first, second)
        t_near = np.maximum(t_near, np.where(parallel, -np.inf, axis_near))
        t_far = np.minimum(t_far, np.where(parallel, np.inf, axis_far))
    hit = (~missed) & (t_far >= t_near) & (t_far > 1e-4)
    times = np.full(count, np.inf, dtype=np.float64)
    chosen = np.where(t_near > 1e-4, t_near, t_far)
    times[hit] = chosen[hit]
    return times


class Scene:
    """A small world of boxes and spheres sampled by ray casting."""

    def __init__(self, primitives: list[Sphere | Box]) -> None:
        self.primitives = primitives

    def scan(self, pose: RoverPose, time_sec: float, model: LidarModel) -> np.ndarray:
        """Return an ``(N, 3)`` cloud in the sensor frame. Misses are omitted."""
        directions = _sensor_directions(model)
        world_directions = directions @ pose.rotation.T
        origin = np.asarray(pose.position, dtype=np.float64)
        best = np.full(directions.shape[0], np.inf, dtype=np.float64)
        for primitive in self.primitives:
            if not primitive.active(time_sec):
                continue
            if isinstance(primitive, Sphere):
                times = _ray_sphere(origin, world_directions, primitive.center_at(time_sec), primitive.radius)
            else:
                times = _ray_box(
                    origin,
                    world_directions,
                    np.asarray(primitive.minimum, dtype=np.float64),
                    np.asarray(primitive.maximum, dtype=np.float64),
                )
            closer = times < best
            best = np.where(closer, times, best)

        rng = np.random.default_rng(model.seed + int(round(time_sec * 1000.0)))
        if model.range_noise_std > 0.0:
            best = best + rng.normal(0.0, model.range_noise_std, size=best.shape)
        keep = np.isfinite(best) & (best >= model.min_range) & (best <= model.max_range)
        if not np.any(keep):
            return np.zeros((0, 3), dtype=np.float64)
        world_points = origin + world_directions[keep] * best[keep, None]
        return (world_points - origin) @ pose.rotation


def demo_scene() -> Scene:
    """Walls, scattered rocks, a moving boulder, and an object that disappears."""
    rocks = [
        Sphere("rock_a", np.array([3.0, 1.2, 0.35]), 0.35),
        Sphere("rock_b", np.array([3.6, -0.8, 0.25]), 0.25),
        Sphere("rock_c", np.array([2.4, -1.4, 0.2]), 0.2),
        Sphere("rock_d", np.array([4.2, 0.4, 0.45]), 0.4),
        Sphere("rock_e", np.array([2.2, 0.6, 0.18]), 0.18),
    ]
    structures = [
        Box("front_wall", np.array([5.0, -2.0, 0.0]), np.array([5.25, 2.0, 2.0])),
        Box("side_wall", np.array([1.0, 2.2, 0.0]), np.array([4.5, 2.45, 1.6])),
        Sphere("mover", np.array([2.8, -1.6, 0.35]), 0.3, velocity=np.array([0.0, 0.35, 0.0])),
        Sphere(
            "temporary",
            np.array([3.4, 1.8, 0.3]),
            0.28,
            active_start=0.0,
            active_end=4.0,
        ),
    ]
    return Scene(rocks + structures)


def _sensor_directions(model: LidarModel) -> np.ndarray:
    half = model.horizontal_fov_deg * 0.5
    full_turn = model.horizontal_fov_deg >= 360.0
    azimuth = np.deg2rad(
        model.horizontal_center_deg
        + np.linspace(-half, half, model.azimuth_bins, endpoint=not full_turn)
    )
    if full_turn:
        azimuth = np.deg2rad(np.linspace(0.0, 360.0, model.azimuth_bins, endpoint=False))
    elevation = np.deg2rad(np.linspace(model.vertical_min_deg, model.vertical_max_deg, model.elevation_bins))
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation, indexing="ij")
    return np.stack(
        [
            np.cos(elevation_grid) * np.cos(azimuth_grid),
            np.cos(elevation_grid) * np.sin(azimuth_grid),
            np.sin(elevation_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
