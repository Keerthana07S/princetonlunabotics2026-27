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
class Mound:
    """A Gaussian hill (positive ``amplitude``) or a crater with a raised rim (negative)."""

    center: tuple[float, float]
    amplitude: float
    sigma: float


@dataclass
class Terrain:
    """Height field ``z = h(x, y)`` built from gentle waves plus mounds and craters.

    Rays are intersected by marching at ``march_step`` meters and interpolating
    the crossing, so slopes much steeper than the step allows are not modelled.
    """

    name: str
    base_height: float = 0.0
    wave_amplitude: float = 0.06
    wave_length: float = 7.0
    mounds: list[Mound] = field(default_factory=list)
    march_step: float = 0.1

    def active(self, time_sec: float) -> bool:
        del time_sec
        return True

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        k = 2.0 * np.pi / self.wave_length
        z = self.base_height + self.wave_amplitude * (np.sin(k * x) * np.cos(0.7 * k * y) + 0.5 * np.sin(1.9 * k * y + 0.4))
        for mound in self.mounds:
            radius_sq = (x - mound.center[0]) ** 2 + (y - mound.center[1]) ** 2
            bowl = np.exp(-radius_sq / (2.0 * mound.sigma * mound.sigma))
            if mound.amplitude >= 0.0:
                z = z + mound.amplitude * bowl
            else:
                rim_radius = 1.6 * mound.sigma
                rim = np.exp(-((np.sqrt(radius_sq) - rim_radius) ** 2) / (2.0 * (0.35 * mound.sigma) ** 2))
                z = z + mound.amplitude * bowl - 0.35 * mound.amplitude * rim
        return z

    def intersect(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along each unit ray to the first terrain crossing, ``inf`` for a miss."""
        times = np.full(directions.shape[0], np.inf, dtype=np.float64)
        candidates = np.flatnonzero(directions[:, 2] < 0.05)
        if candidates.size == 0:
            return times
        steps = np.arange(1, int(np.ceil(max_range / self.march_step)) + 1, dtype=np.float64) * self.march_step
        rays = directions[candidates]
        samples = origin[None, None, :] + rays[:, None, :] * steps[None, :, None]
        below = samples[:, :, 2] <= self.height(samples[:, :, 0], samples[:, :, 1])
        hit = below.any(axis=1)
        if not np.any(hit):
            return times
        first = np.argmax(below, axis=1)
        rows = np.flatnonzero(hit)
        index = first[rows]
        t_after = steps[index]
        t_before = np.where(index > 0, steps[np.maximum(index - 1, 0)], 0.0)
        ray = rays[rows]
        before_point = origin + ray * t_before[:, None]
        after_point = origin + ray * t_after[:, None]
        gap_before = before_point[:, 2] - self.height(before_point[:, 0], before_point[:, 1])
        gap_after = after_point[:, 2] - self.height(after_point[:, 0], after_point[:, 1])
        denominator = gap_before - gap_after
        safe = np.where(np.abs(denominator) > 1e-9, denominator, 1.0)
        fraction = np.clip(np.where(np.abs(denominator) > 1e-9, gap_before / safe, 1.0), 0.0, 1.0)
        times[candidates[rows]] = t_before + fraction * (t_after - t_before)
        return times


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


def line_pose(time_sec: float, speed: float, max_distance: float = float("inf")) -> RoverPose:
    """Drive forward along +x at a constant speed and stop after ``max_distance``."""
    distance = min(speed * max(time_sec, 0.0), max_distance)
    return RoverPose(position=np.array([distance, 0.0, 0.0]), yaw=0.0)


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
    """A world of terrain, boxes, and spheres sampled by ray casting.

    ``sensor_height`` is the LiDAR height above the terrain. With terrain the
    sensor rides ``sensor_height`` above ``h(x, y)``; without it the sensor
    stays at the pose's own z.
    """

    def __init__(self, primitives: list[Sphere | Box | Terrain], sensor_height: float = 0.0) -> None:
        self.primitives = primitives
        self.sensor_height = float(sensor_height)

    @property
    def terrain(self) -> Terrain | None:
        for primitive in self.primitives:
            if isinstance(primitive, Terrain):
                return primitive
        return None

    def sensor_pose(self, pose: RoverPose) -> RoverPose:
        """Place the sensor on the terrain surface plus ``sensor_height``."""
        terrain = self.terrain
        if terrain is None:
            return pose
        position = np.asarray(pose.position, dtype=np.float64).copy()
        position[2] = float(terrain.height(position[0], position[1])) + self.sensor_height
        return RoverPose(position=position, yaw=pose.yaw)

    def scan(self, pose: RoverPose, time_sec: float, model: LidarModel) -> np.ndarray:
        """Return an ``(N, 3)`` cloud in the sensor frame. Misses are omitted.

        ``pose`` is the sensor pose; call :meth:`sensor_pose` first to place it on terrain.
        """
        directions = _sensor_directions(model)
        world_directions = directions @ pose.rotation.T
        origin = np.asarray(pose.position, dtype=np.float64)
        best = np.full(directions.shape[0], np.inf, dtype=np.float64)
        for primitive in self.primitives:
            if not primitive.active(time_sec):
                continue
            if isinstance(primitive, Terrain):
                times = primitive.intersect(origin, world_directions, model.max_range)
            elif isinstance(primitive, Sphere):
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


DEMO_PATH_LENGTH = 9.0
"""Meters the demo rover drives along +x before stopping short of the far wall."""


def demo_scene(sensor_height: float = 0.6) -> Scene:
    """Cratered terrain, rocks, walls, a moving boulder, and objects that disappear and appear.

    The rover drives along +x from the origin. Timed objects:
    ``mover`` crosses the path, ``temporary`` vanishes at 12 s, and
    ``arrival`` appears at 10 s.
    """
    terrain = Terrain(
        "terrain",
        mounds=[
            Mound((2.5, -2.4), 0.35, 0.8),
            Mound((5.5, 2.6), 0.45, 1.1),
            Mound((8.5, -3.0), 0.4, 0.9),
            Mound((4.0, 3.8), -0.35, 0.9),
            Mound((7.0, -1.8), -0.3, 0.7),
            Mound((10.5, 2.0), -0.4, 1.0),
            Mound((1.0, 3.0), 0.25, 0.6),
            Mound((11.5, -2.2), 0.3, 0.8),
        ],
    )

    def on_ground(x: float, y: float, radius: float) -> np.ndarray:
        return np.array([x, y, float(terrain.height(x, y)) + 0.6 * radius])

    rock_layout = [
        ("rock_a", 3.0, 1.2, 0.35),
        ("rock_b", 3.6, -0.9, 0.25),
        ("rock_c", 2.2, -1.4, 0.2),
        ("rock_d", 4.6, 0.8, 0.45),
        ("rock_e", 1.8, 0.7, 0.18),
        ("rock_f", 6.2, -1.1, 0.3),
        ("rock_g", 7.4, 1.4, 0.4),
        ("rock_h", 8.8, -0.7, 0.22),
        ("rock_i", 9.9, 1.1, 0.35),
        ("rock_j", 11.2, -1.3, 0.5),
        ("rock_k", 5.4, -2.6, 0.3),
        ("rock_l", 0.6, -1.8, 0.28),
    ]
    rocks = [Sphere(name, on_ground(x, y, radius), radius) for name, x, y, radius in rock_layout]
    structures = [
        Box("far_wall", np.array([13.5, -4.0, -0.5]), np.array([13.8, 4.0, 2.2])),
        Box("side_wall", np.array([1.5, 4.4, -0.5]), np.array([9.5, 4.7, 1.6])),
        Box("block", np.array([6.6, -3.9, -0.5]), np.array([7.6, -3.1, 1.1])),
        Sphere("mover", on_ground(5.0, -3.2, 0.3), 0.3, velocity=np.array([0.0, 0.3, 0.0])),
        Sphere("temporary", on_ground(3.4, 2.0, 0.3), 0.3, active_start=0.0, active_end=12.0),
        Box(
            "arrival",
            np.array([8.0, -2.8, -0.5]),
            np.array([8.6, -2.2, 0.9]),
            active_start=10.0,
        ),
    ]
    return Scene([terrain] + rocks + structures, sensor_height=sensor_height)


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
