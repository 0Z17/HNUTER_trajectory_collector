#!/usr/bin/env python3
"""Free-flight obstacle geometry, route certificates, and condition encoding.

The generated environment JSON follows the repository's box convention.  In
particular, non-floor collision boxes can be passed directly to
``se3_diffusion._obstacle_tokens`` as 10-D ``[xyz, size_xyz, quaternion_wxyz]``
cross-attention tokens.  Task orchestration, HTTP, and CLI concerns live in
``expert_trajectory_collector``; ``main`` remains only as a legacy launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import random
from typing import Any, Callable
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
MAX_OBSTACLES = 32
MAX_FLIGHT_ROLL_DEG = 40.0
MAX_FLIGHT_PITCH_DEG = 70.0
WORKSPACE_BOUNDS = {"min": [-4.0, -4.0, 0.0], "max": [4.0, 4.0, 4.0]}
SAMPLING_BOUNDS = {"min": [-3.3, -3.3, 0.65], "max": [3.3, 3.3, 3.5]}
URDF_PATH = (
    PROJECT_DIR
    / "expert_trajectory_collector/assets/HDJQR-0102-0055.SLDASM.urdf"
)


@dataclass(frozen=True)
class RobotPrimitive:
    name: str
    kind: str
    local_position: tuple[float, float, float]
    local_quaternion_wxyz: tuple[float, float, float, float]
    half_extents: tuple[float, float, float]


@dataclass(frozen=True)
class RobotEnvelope:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    safety_margin: float
    collision_names: tuple[str, ...]
    primitives: tuple[RobotPrimitive, ...]


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _rpy_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _numbers(raw: str | None, count: int, default: float = 0.0) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split()) if raw else (default,) * count
    if len(values) != count:
        raise ValueError(f"expected {count} values, received {len(values)}")
    return values


def load_robot_envelope(path: Path = URDF_PATH) -> RobotEnvelope:
    """Derive a conservative local AABB from active base-link URDF collisions."""
    root = ET.parse(path).getroot()
    link = next((item for item in root.findall("link") if item.get("name") == "base_link"), None)
    if link is None:
        raise ValueError(f"URDF {path} has no base_link")
    lows, highs, names, primitives = [], [], [], []
    for index, collision in enumerate(link.findall("collision")):
        origin = collision.find("origin")
        xyz = np.asarray(_numbers(origin.get("xyz") if origin is not None else None, 3))
        rpy = _numbers(origin.get("rpy") if origin is not None else None, 3)
        rotation = _rpy_matrix(rpy)
        geometry = collision.find("geometry")
        shape = next(iter(geometry), None) if geometry is not None else None
        if shape is None:
            continue
        if shape.tag == "box":
            half = np.asarray(_numbers(shape.get("size"), 3)) / 2
        elif shape.tag == "sphere":
            half = np.full(3, float(shape.get("radius", "0")))
        elif shape.tag == "cylinder":
            radius = float(shape.get("radius", "0"))
            half = np.asarray([radius, radius, float(shape.get("length", "0")) / 2])
        else:
            continue
        world_half = np.abs(rotation) @ half
        lows.append(xyz - world_half)
        highs.append(xyz + world_half)
        names.append(collision.get("name", f"collision_{index}"))
        primitives.append(RobotPrimitive(
            names[-1], shape.tag, tuple(float(value) for value in xyz),
            tuple(float(value) for value in _rpy_quaternion(*rpy)),
            tuple(float(value) for value in half),
        ))
    if not lows:
        raise ValueError(f"URDF {path} has no supported active base-link collisions")
    minimum = np.min(np.stack(lows), axis=0)
    maximum = np.max(np.stack(highs), axis=0)
    size = maximum - minimum
    margin = min(0.15, max(0.05, 0.1 * float(max(size[:2]))))
    return RobotEnvelope(
        tuple(float(value) for value in minimum),
        tuple(float(value) for value in maximum),
        tuple(float(value) for value in (minimum + maximum) / 2),
        tuple(float(value) for value in size),
        margin, tuple(names), tuple(primitives),
    )


ROBOT = load_robot_envelope()


@dataclass(frozen=True)
class SceneParameters:
    family: str = "staggered_corridor"
    seed: int = 1
    obstacle_count: int = 7
    size_min: float = 0.45
    size_max: float = 1.00
    gap_width: float = 1.65
    global_yaw_deg: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SceneParameters":
        values: dict[str, Any] = {}
        for field in cls.__dataclass_fields__.values():
            if field.name not in raw:
                continue
            values[field.name] = (
                str(raw[field.name]) if field.name == "family"
                else int(raw[field.name]) if field.name in {"seed", "obstacle_count"}
                else float(raw[field.name])
            )
        params = cls(**values)
        if params.family not in FAMILY_BUILDERS:
            raise ValueError(f"unknown scene family {params.family!r}")
        if not 1 <= params.obstacle_count <= MAX_OBSTACLES:
            raise ValueError(f"obstacle_count must be in [1, {MAX_OBSTACLES}]")
        minimum = FAMILY_MINIMUMS[params.family]
        if params.obstacle_count < minimum:
            raise ValueError(
                f"{params.family} needs at least {minimum} obstacles to retain its structure"
            )
        if not 0.15 <= params.size_min <= params.size_max <= 3.5:
            raise ValueError("sizes must satisfy 0.15 <= size_min <= size_max <= 3.5")
        if not 0.65 <= params.gap_width <= 3.2:
            raise ValueError("gap_width must be in [0.65, 3.2]")
        if params.family in {
            "narrow_passage", "orientation_sensitive_passage",
            "wall_protrusion_bracket", "frame_doorway",
        }:
            useful_min, useful_max = passage_limits(params.family)
            if not useful_min <= params.gap_width <= useful_max:
                raise ValueError(
                    f"{params.family} gap must be within the URDF-feasible range "
                    f"[{useful_min:.3f}, {useful_max:.3f}] m"
                )
        if abs(params.translate_x) > 0.75 or abs(params.translate_y) > 0.75:
            raise ValueError("global translation must stay within +/-0.75 m")
        return params


def passage_limits(family: str) -> tuple[float, float]:
    """Return useful aperture-width bounds derived from the URDF envelope."""
    horizontal_min = min(ROBOT.size[0], ROBOT.size[1])
    horizontal_max = max(ROBOT.size[0], ROBOT.size[1])
    if family == "orientation_sensitive_passage":
        aligned_width = ROBOT.size[1] + 2 * ROBOT.safety_margin
        return aligned_width + 0.08, aligned_width + 0.18
    return (
        horizontal_min + 2 * ROBOT.safety_margin + 0.025,
        min(2.05, horizontal_max + 2 * ROBOT.safety_margin + 0.42),
    )


def sample_scene_parameters(raw: dict[str, Any]) -> tuple[SceneParameters, dict[str, Any]]:
    """Resolve UI ranges into one deterministic parameter sample."""
    family = str(raw.get("family", SceneParameters.family))
    seed = int(raw.get("seed", 1))
    if family not in FAMILY_BUILDERS:
        raise ValueError(f"unknown scene family {family!r}")
    if not raw.get("sample_ranges", False):
        mapping = dict(raw)
        if family == "orientation_sensitive_passage" and "gap_width" not in mapping:
            limits = passage_limits(family)
            mapping["gap_width"] = sum(limits) / 2
        return SceneParameters.from_mapping(mapping), {}
    rng = random.Random(f"scene-parameters/{seed}/{family}")
    count_min = max(FAMILY_MINIMUMS[family], int(raw.get("obstacle_count_min", 5)))
    count_max = int(raw.get("obstacle_count_max", 12))
    if count_min > count_max or count_max > MAX_OBSTACLES:
        raise ValueError(
            f"obstacle count range must satisfy {FAMILY_MINIMUMS[family]} <= min <= max <= {MAX_OBSTACLES}"
        )
    useful_gap_min, useful_gap_max = passage_limits(family)
    requested_gap_min = float(raw.get("gap_width_min", useful_gap_min))
    requested_gap_max = float(raw.get("gap_width_max", useful_gap_max))
    gap_min = max(useful_gap_min, requested_gap_min)
    gap_max = min(useful_gap_max, requested_gap_max)
    if gap_min > gap_max:
        raise ValueError(
            f"gap range has no URDF-feasible width; use approximately {useful_gap_min:.2f}–{useful_gap_max:.2f} m"
        )
    yaw_min = float(raw.get("global_yaw_min", -35.0))
    yaw_max = float(raw.get("global_yaw_max", 35.0))
    translation_max = float(raw.get("translation_max", 0.3))
    if yaw_min > yaw_max or not 0 <= translation_max <= 0.55:
        raise ValueError("invalid global transform range")
    resolved = {
        "family": family,
        "seed": seed,
        "obstacle_count": rng.randint(count_min, count_max),
        "size_min": float(raw.get("size_min", 0.35)),
        "size_max": float(raw.get("size_max", 1.00)),
        "gap_width": rng.uniform(gap_min, gap_max),
        "global_yaw_deg": rng.uniform(yaw_min, yaw_max),
        "translate_x": rng.uniform(-translation_max, translation_max),
        "translate_y": rng.uniform(-translation_max, translation_max),
    }
    ranges = {
        "obstacle_count": [count_min, count_max],
        "gap_width_m": [gap_min, gap_max],
        "global_yaw_deg": [yaw_min, yaw_max],
        "translation_abs_max_m": translation_max,
        "size_m": [resolved["size_min"], resolved["size_max"]],
    }
    return SceneParameters.from_mapping(resolved), ranges


def yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _box(
    identifier: str, center: tuple[float, float, float],
    size: tuple[float, float, float], *, yaw: float = 0.0,
    role: str = "obstacle", family: str = "", assembly_group: str | None = None,
    quaternion_wxyz: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    result = {
        "id": identifier,
        "type": "box",
        "pose": {
            "position": [round(float(value), 6) for value in center],
            "quaternion_wxyz": [
                round(float(value), 9)
                for value in (quaternion_wxyz if quaternion_wxyz is not None else yaw_quaternion(yaw))
            ],
        },
        "size_xyz": [round(float(value), 6) for value in size],
        "collision": True,
        "visual": True,
        "static": True,
        "role": role,
        "family": family,
    }
    if assembly_group is not None:
        result["assembly_group"] = assembly_group
    rotation = _rpy_matrix((0.0, 0.0, yaw)) if quaternion_wxyz is None else None
    if rotation is None:
        q = np.asarray(quaternion_wxyz, dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        w, x, y, z = q
        rotation = np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
    world_half = np.abs(rotation) @ (np.asarray(size, dtype=np.float64) / 2)
    if float(center[2]) - float(world_half[2]) <= 1e-5:
        result["physical_support"] = "ground"
    return result


def rpy_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    return _rpy_quaternion(roll, pitch, yaw)


def quaternion_multiply(first: list[float], second: list[float]) -> list[float]:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    result = np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)
    result /= max(float(np.linalg.norm(result)), 1e-12)
    return result.tolist()


def quaternion_matrix(quaternion: list[float] | tuple[float, ...]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quaternion_roll_pitch_degrees(
    quaternion: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[float, float]:
    """Return intrinsic ZYX roll/pitch angles for a wxyz quaternion."""
    rotation = quaternion_matrix(quaternion)
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.asin(min(1.0, max(-1.0, -float(rotation[2, 0]))))
    return math.degrees(roll), math.degrees(pitch)


def attitude_is_within_flight_limits(
    quaternion: list[float] | tuple[float, ...] | np.ndarray,
    tolerance_deg: float = 1e-6,
) -> bool:
    roll, pitch = quaternion_roll_pitch_degrees(quaternion)
    return (
        abs(roll) <= MAX_FLIGHT_ROLL_DEG + tolerance_deg
        and abs(pitch) <= MAX_FLIGHT_PITCH_DEG + tolerance_deg
    )


def _obb_overlap(
    center_a: np.ndarray, rotation_a: np.ndarray, half_a: np.ndarray,
    center_b: np.ndarray, rotation_b: np.ndarray, half_b: np.ndarray,
) -> bool:
    """Separating-axis test for two 3-D oriented boxes."""
    relative = rotation_a.T @ rotation_b
    absolute = np.abs(relative) + 1e-9
    translation = rotation_a.T @ (center_b - center_a)
    for axis in range(3):
        if abs(translation[axis]) >= half_a[axis] + float(absolute[axis] @ half_b):
            return False
    for axis in range(3):
        if abs(float(translation @ relative[:, axis])) >= float(half_a @ absolute[:, axis]) + half_b[axis]:
            return False
    for first in range(3):
        for second in range(3):
            radius_a = half_a[(first + 1) % 3] * absolute[(first + 2) % 3, second] + half_a[(first + 2) % 3] * absolute[(first + 1) % 3, second]
            radius_b = half_b[(second + 1) % 3] * absolute[first, (second + 2) % 3] + half_b[(second + 2) % 3] * absolute[first, (second + 1) % 3]
            projected = abs(
                translation[(first + 2) % 3] * relative[(first + 1) % 3, second]
                - translation[(first + 1) % 3] * relative[(first + 2) % 3, second]
            )
            if projected >= radius_a + radius_b:
                return False
    return True


def _obstacle_obb(obstacle: dict[str, Any], padding: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(obstacle["pose"]["position"], dtype=np.float64)
    rotation = quaternion_matrix(obstacle["pose"]["quaternion_wxyz"])
    half = np.asarray(obstacle["size_xyz"], dtype=np.float64) / 2 + padding
    return center, rotation, half


def obstacles_overlap(
    first: dict[str, Any], second: dict[str, Any], padding: float = 0.0,
) -> bool:
    return _obb_overlap(*_obstacle_obb(first, padding), *_obstacle_obb(second, padding))


def _robot_obb(pose: list[float], safety_margin: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = quaternion_matrix(pose[3:7])
    reference = np.asarray(pose[:3], dtype=np.float64)
    center = reference + rotation @ np.asarray(ROBOT.center, dtype=np.float64)
    margin = ROBOT.safety_margin if safety_margin is None else safety_margin
    half = np.asarray(ROBOT.size, dtype=np.float64) / 2 + margin
    return center, rotation, half


def _robot_primitive_obbs(
    pose: list[float], safety_margin: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    vehicle_rotation = quaternion_matrix(pose[3:7])
    reference = np.asarray(pose[:3], dtype=np.float64)
    margin = ROBOT.safety_margin if safety_margin is None else safety_margin
    output = []
    for primitive in ROBOT.primitives:
        local_position = np.asarray(primitive.local_position, dtype=np.float64)
        local_rotation = quaternion_matrix(primitive.local_quaternion_wxyz)
        output.append((
            reference + vehicle_rotation @ local_position,
            vehicle_rotation @ local_rotation,
            np.asarray(primitive.half_extents, dtype=np.float64) + margin,
        ))
    return output


def robot_projection_size(roll_radians: float) -> tuple[float, float]:
    """Conservative Y/Z projection of every URDF primitive at one roll."""
    pose = [0.0, 0.0, 0.0, *rpy_quaternion(roll_radians, 0.0, 0.0)]
    lows, highs = [], []
    for center, rotation, half in _robot_primitive_obbs(pose):
        world_half = np.abs(rotation) @ half
        lows.append(center - world_half)
        highs.append(center + world_half)
    minimum = np.min(np.stack(lows), axis=0)
    maximum = np.max(np.stack(highs), axis=0)
    return float(maximum[1] - minimum[1]), float(maximum[2] - minimum[2])


def pose_is_free(
    pose: list[float], obstacles: list[dict[str, Any]],
    safety_margin: float | None = None,
) -> bool:
    return not any(
        _obb_overlap(*robot, *_obstacle_obb(obstacle))
        for robot in _robot_primitive_obbs(pose, safety_margin)
        for obstacle in obstacles
    )


def _interpolate_route(route: list[list[float]], spacing: float = 0.09) -> list[list[float]]:
    dense: list[list[float]] = []
    for start, goal in zip(route, route[1:]):
        distance = float(np.linalg.norm(np.asarray(goal[:3]) - np.asarray(start[:3])))
        start_q = np.asarray(start[3:7], dtype=np.float64)
        goal_q = np.asarray(goal[3:7], dtype=np.float64)
        if float(start_q @ goal_q) < 0:
            goal_q *= -1
        angular_distance = 2 * math.acos(
            min(1.0, max(-1.0, abs(float(start_q @ goal_q))))
        )
        steps = max(
            1, math.ceil(distance / spacing),
            math.ceil(angular_distance / math.radians(5.0)),
        )
        for step in range(steps):
            alpha = step / steps
            position = (1 - alpha) * np.asarray(start[:3]) + alpha * np.asarray(goal[:3])
            quaternion = (1 - alpha) * start_q + alpha * goal_q
            quaternion /= np.linalg.norm(quaternion)
            dense.append([*position.tolist(), *quaternion.tolist()])
    dense.append(route[-1])
    return dense


def route_is_free(
    route: list[list[float]], obstacles: list[dict[str, Any]],
    safety_margin: float | None = None,
) -> bool:
    return all(
        pose_is_free(pose, obstacles, safety_margin)
        for pose in _interpolate_route(route)
    )


def _wall_segments(
    prefix: str, x: float, gap_center: float, gap: float, *,
    thickness: float = 0.35, height: float = 3.6, yaw: float = 0.0,
    family: str, assembly_group: str | None = None,
) -> list[dict[str, Any]]:
    extent = 3.05
    low_end, high_start = gap_center - gap / 2, gap_center + gap / 2
    pieces = []
    if low_end > -extent:
        length = low_end + extent
        pieces.append(_box(
            f"{prefix}_low", (x, -extent + length / 2, height / 2),
            (thickness, length, height), yaw=yaw, family=family,
            assembly_group=assembly_group,
        ))
    if high_start < extent:
        length = extent - high_start
        pieces.append(_box(
            f"{prefix}_high", (x, high_start + length / 2, height / 2),
            (thickness, length, height), yaw=yaw, family=family,
            assembly_group=assembly_group,
        ))
    for piece in pieces:
        piece["aperture_center_y"] = round(float(gap_center), 6)
        piece["aperture_width"] = round(float(gap), 6)
        piece["wall_x"] = round(float(x), 6)
    return pieces


def _sparse_clutter(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    # Keep one randomly positioned, compact but flight-relevant anchor.  The
    # remaining boxes are still sparse filler, but route modes now describe
    # genuine ways around an obstacle instead of arbitrary empty-space arcs.
    height = rng.uniform(1.95, 2.55)
    anchor = _box(
        "clutter_anchor",
        (rng.uniform(-0.35, 0.35), rng.uniform(-0.30, 0.30), height / 2),
        (rng.uniform(0.48, 0.82), rng.uniform(1.10, 1.65), height),
        yaw=rng.uniform(-0.18, 0.18), family=p.family,
    )
    anchor["route_separator"] = True
    return [anchor]


def _central_block(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    side = rng.choice((-1, 1))
    # Keep both lateral centre corridors wider than the conservative URDF
    # envelope.  The former 2.75--3.65 m depth frequently left only one
    # flyable side after safety inflation in the 6.6 m sampling span.
    width = rng.uniform(0.65, 1.00)
    depth = rng.uniform(1.90, 2.60)
    # Tall enough to invalidate the straight mid-height connection, while
    # leaving a deliberately narrow but URDF-feasible over route in many
    # seeds.  Near-ceiling blockers remain represented by the wall families.
    height = rng.uniform(2.15, 2.65)
    result = [_box(
        "central_block", (rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), height / 2),
        (width, depth, height), yaw=side * rng.uniform(0.02, 0.12), family=p.family,
    )]
    result[0]["route_separator"] = True
    return result


def _multi_homotopy(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    # A supported transverse beam admits four meaningful choices for the same
    # endpoint pair: above, below, left and right.  Thin posts make the
    # elevated member physically constructible without closing the underpass.
    width = rng.uniform(0.34, 0.50)
    depth = rng.uniform(2.05, 2.55)
    height = rng.uniform(0.42, 0.58)
    beam_z = rng.uniform(1.90, 2.10)
    center_x = rng.uniform(-0.18, 0.18)
    center_y = rng.uniform(-0.18, 0.18)
    group = "four_way_supported_separator"
    separator = _box(
        "separator_beam", (center_x, center_y, beam_z),
        (width, depth, height), yaw=rng.uniform(-0.06, 0.06),
        family=p.family, assembly_group=group,
    )
    separator["homotopy_separator"] = True
    separator["physical_support"] = "separator_support_columns"
    beam_bottom = beam_z - height / 2
    post_size = rng.uniform(0.13, 0.18)
    support_offset = depth / 2 - post_size / 2
    result = [
        separator,
        _box(
            "separator_support_left",
            (center_x, center_y + support_offset, beam_bottom / 2),
            (post_size, post_size, beam_bottom), family=p.family,
            role="structural_support", assembly_group=group,
        ),
        _box(
            "separator_support_right",
            (center_x, center_y - support_offset, beam_bottom / 2),
            (post_size, post_size, beam_bottom), family=p.family,
            role="structural_support", assembly_group=group,
        ),
    ]
    return result[:p.obstacle_count]


def _narrow_passage(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    gap_center = rng.uniform(-0.5, 0.5)
    wall_x = rng.uniform(-0.3, 0.3)
    result = _wall_segments(
        "narrow_wall", wall_x, gap_center, p.gap_width,
        thickness=rng.uniform(0.3, 0.52), height=4.0, family=p.family,
    )
    return result[:p.obstacle_count]


def _orientation_passage(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    gap = p.gap_width
    gap_center = rng.uniform(-0.35, 0.35)
    wall_x = rng.uniform(-0.22, 0.22)
    thickness = rng.uniform(0.24, 0.36)
    target_roll = math.radians(
        rng.uniform(25.0, MAX_FLIGHT_ROLL_DEG)
    ) * rng.choice((-1, 1))
    quaternion = rpy_quaternion(target_roll, 0.0, 0.0)
    center = np.asarray([wall_x, gap_center, rng.uniform(1.9, 2.1)])
    u_axis = np.asarray([0.0, math.cos(target_roll), math.sin(target_roll)])
    v_axis = np.asarray([0.0, -math.sin(target_roll), math.cos(target_roll)])
    outer_u = 3.5
    outer_v = 1.9
    aligned_height = robot_projection_size(0.0)[1]
    clear_height = aligned_height + rng.uniform(0.10, 0.16)
    side_length = (outer_u - gap) / 2
    vertical_piece = (outer_v - clear_height) / 2

    def framed_box(
        identifier: str, position: np.ndarray, size: tuple[float, float, float]
    ) -> dict[str, Any]:
        obstacle = _box(
            identifier, tuple(position), size, family=p.family,
            assembly_group="orientation_gate", quaternion_wxyz=quaternion,
        )
        obstacle.update({
            "aperture_center_y": round(gap_center, 6),
            "wall_x": round(wall_x, 6),
            "aperture_center_xyz": [round(float(value), 6) for value in center],
            "aperture_width": round(gap, 6),
            "aperture_height": round(clear_height, 6),
            "required_roll_deg": round(math.degrees(target_roll), 3),
        })
        return obstacle

    result = [
        framed_box("slot_low", center - u_axis * (gap / 2 + side_length / 2), (thickness, side_length, outer_v)),
        framed_box("slot_high", center + u_axis * (gap / 2 + side_length / 2), (thickness, side_length, outer_v)),
        framed_box("slot_floor_lip", center - v_axis * (clear_height / 2 + vertical_piece / 2), (thickness, gap, vertical_piece)),
        framed_box("slot_ceiling_lip", center + v_axis * (clear_height / 2 + vertical_piece / 2), (thickness, gap, vertical_piece)),
    ]
    supports = []
    for side_name, bar in zip(("low", "high"), result[:2]):
        bar_center, bar_rotation, bar_half = _obstacle_obb(bar)
        bottom = max(0.12, float(bar_center[2] - (np.abs(bar_rotation) @ bar_half)[2]))
        support = _box(
            f"slot_{side_name}_support",
            (float(bar_center[0]), float(bar_center[1]), bottom / 2),
            (0.16, 0.16, bottom), family=p.family,
            role="structural_support", assembly_group="orientation_gate",
        )
        support["supports_obstacle"] = bar["id"]
        supports.append(support)
    for obstacle in result:
        obstacle["physical_support"] = "orientation_gate_columns"
    result.extend(supports)
    return result[:p.obstacle_count]


def _wall_protrusion(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    side = rng.choice((-1, 1))
    gap_center = side * rng.uniform(0.55, 0.85)
    wall_x = rng.uniform(0.45, 0.70)
    result = _wall_segments(
        "bracket_wall", wall_x, gap_center, p.gap_width,
        thickness=rng.uniform(0.24, 0.34), height=4.0, family=p.family,
        assembly_group="bracket_assembly",
    )
    # Build the bracket on the aperture centreline.  Its grounded vertical tip
    # blocks the straight portal approach, while the low/high arms connect the
    # tip back to the wall and leave a physically buildable middle-height bay.
    # The previous opposite-side placement was decorative: removing it did not
    # change the route at all.
    bracket_y = gap_center
    # Leave enough longitudinal room between the tip and the wall for the
    # complete vehicle envelope to finish its lateral dogleg before the portal.
    tip_x = rng.uniform(-1.72, -1.58)
    arm_center_x = 0.5 * (tip_x + wall_x)
    arm_length = wall_x - tip_x + 0.10
    result.extend([
        _box("bracket_arm_low", (arm_center_x, bracket_y, 0.66), (arm_length, 0.26, 0.26), family=p.family, assembly_group="bracket_assembly"),
        _box("bracket_arm_high", (arm_center_x, bracket_y, 2.94), (arm_length, 0.26, 0.26), family=p.family, assembly_group="bracket_assembly"),
        _box("bracket_tip", (tip_x, bracket_y, 1.80), (0.28, 0.28, 3.60), family=p.family, assembly_group="bracket_assembly"),
    ])
    for obstacle in result:
        obstacle["route_relevant"] = True
        obstacle["route_constraint"] = "wall_portal_bracket_dogleg"
        if obstacle.get("physical_support") != "ground":
            obstacle["physical_support"] = "bracket_assembly"
    result[-1].update({
        "bracket_tip_x": round(float(tip_x), 6),
        "bracket_center_y": round(float(bracket_y), 6),
        "wall_x": round(float(wall_x), 6),
        "aperture_center_y": round(float(gap_center), 6),
        "aperture_width": round(float(p.gap_width), 6),
    })
    return result[:p.obstacle_count]


def _frame_doorway(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    gap = p.gap_width
    post_width = rng.uniform(0.24, 0.38)
    # Keep enough headroom for the through route while retaining usable space
    # above the header for a second, genuinely vertical route mode.
    clear_height = rng.uniform(2.00, 2.32)
    thickness = rng.uniform(0.24, 0.36)
    gap_center = rng.uniform(-0.35, 0.35)
    result = [
        _box("frame_left", (0, gap_center - (gap + post_width) / 2, clear_height / 2), (thickness, post_width, clear_height), family=p.family),
        _box("frame_right", (0, gap_center + (gap + post_width) / 2, clear_height / 2), (thickness, post_width, clear_height), family=p.family),
        _box("frame_header", (0, gap_center, clear_height + 0.205), (thickness, gap + 2 * post_width, 0.4), family=p.family),
    ]
    for obstacle in result:
        obstacle["aperture_center_y"] = round(float(gap_center), 6)
        obstacle["aperture_width"] = round(float(gap), 6)
        obstacle["wall_x"] = 0.0
        obstacle["aperture_center_z"] = round(float(clear_height / 2), 6)
    # The header is the outer separator for the above/left/right route modes.
    # The through mode is handled separately by its aperture cross-section.
    result[-1]["route_separator"] = True
    result[-1]["physical_support"] = "doorway_posts"
    return result[:p.obstacle_count]


def _pillar_wall(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    wall_height = rng.uniform(2.0, 2.85)
    result = [_box("short_wall", (rng.uniform(0.40, 0.75), rng.uniform(-0.20, 0.20), wall_height / 2), (rng.uniform(0.28, 0.40), rng.uniform(1.65, 2.05), wall_height), yaw=rng.uniform(-0.07, 0.07), family=p.family)]
    result[0]["route_separator"] = True
    # Keep the semantic pillar near the short wall so the pair forms one
    # central separator instead of accidentally sealing one lateral corridor.
    positions = [(-0.72, rng.uniform(-0.28, 0.28))]
    for index, (x, y) in enumerate(positions):
        radius = rng.uniform(max(0.26, p.size_min * 0.65), min(0.55, p.size_max))
        result.append(_box(
            f"pillar_{index:02d}", (x, y, 1.65), (radius, radius, 3.3),
            yaw=rng.uniform(-0.2, 0.2), family=p.family,
        ))
    return result[:min(p.obstacle_count, 2)]


def _staggered_corridor(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    result = []
    count = min(p.obstacle_count, rng.randint(2, min(5, p.obstacle_count)))
    phase = rng.choice((-1, 1))
    for index in range(count):
        x = -2.35 + index * (4.7 / max(1, count - 1)) + rng.uniform(-0.12, 0.12)
        side = phase * (-1 if index % 2 else 1)
        depth = rng.uniform(max(0.52, p.size_min), min(1.25, p.size_max + 0.28))
        y = side * (2.2 - depth * 0.25)
        height = rng.uniform(2.0, 3.6)
        result.append(_box(
            f"stagger_{index:02d}", (x, y, height / 2),
            (rng.uniform(0.28, 0.50), depth, height),
            yaw=side * rng.uniform(0.0, 0.25), family=p.family,
        ))
    return result


def _mixed_industrial(p: SceneParameters, rng: random.Random) -> list[dict[str, Any]]:
    mirror = rng.choice((-1, 1))
    offset_y = rng.uniform(-0.10, 0.10)
    machine_y = mirror * rng.uniform(0.52, 0.68) + offset_y
    control_y = -mirror * rng.uniform(0.52, 0.68) + offset_y
    machine_x = rng.uniform(-1.58, -1.42)
    control_x = rng.uniform(1.08, 1.24)
    beam_z = rng.uniform(2.24, 2.38)
    beam_height = 0.28
    post_height = beam_z - beam_height / 2
    pipe_group = "supported_pipe_assembly"
    candidates = [
        _box("machine_base", (machine_x, machine_y, 0.45), (0.86, 1.18, 0.90), yaw=mirror * 0.08, family=p.family, assembly_group="machine_assembly"),
        _box("machine_tower", (machine_x, machine_y, 1.95), (0.58, 1.02, 2.10), yaw=mirror * 0.08, family=p.family, assembly_group="machine_assembly"),
        _box("control_box", (control_x, control_y, 1.65), (0.66, 1.18, 3.30), yaw=-mirror * 0.08, family=p.family),
        _box("supported_pipe", (0.0, offset_y, beam_z), (0.34, 5.00, beam_height), yaw=0.0, family=p.family, assembly_group=pipe_group),
        _box("pipe_support_a", (0.0, offset_y - 2.40, post_height / 2), (0.18, 0.18, post_height), family=p.family, role="structural_support", assembly_group=pipe_group),
        _box("pipe_support_b", (0.0, offset_y + 2.40, post_height / 2), (0.18, 0.18, post_height), family=p.family, role="structural_support", assembly_group=pipe_group),
    ]
    candidates[1]["physical_support"] = "machine_base"
    candidates[3]["physical_support"] = "support_columns"
    for obstacle in candidates:
        obstacle["route_relevant"] = True
        obstacle["route_constraint"] = "industrial_winding_underpass"
    return candidates[:p.obstacle_count]


FAMILY_BUILDERS: dict[str, Callable[[SceneParameters, random.Random], list[dict[str, Any]]]] = {
    "sparse_obb_clutter": _sparse_clutter,
    "central_block": _central_block,
    "multi_homotopy": _multi_homotopy,
    "narrow_passage": _narrow_passage,
    "orientation_sensitive_passage": _orientation_passage,
    "wall_protrusion_bracket": _wall_protrusion,
    "frame_doorway": _frame_doorway,
    "pillar_wall": _pillar_wall,
    "staggered_corridor": _staggered_corridor,
    "mixed_industrial": _mixed_industrial,
}

FAMILY_MINIMUMS = {
    "sparse_obb_clutter": 1,
    "central_block": 1,
    "multi_homotopy": 3,
    "narrow_passage": 2,
    "orientation_sensitive_passage": 6,
    "wall_protrusion_bracket": 5,
    "frame_doorway": 3,
    "pillar_wall": 2,
    "staggered_corridor": 2,
    "mixed_industrial": 6,
}


FAMILY_LABELS = {
    "sparse_obb_clutter": "随机稀疏 OBB clutter",
    "central_block": "中央阻挡 / 必须绕行",
    "multi_homotopy": "四通道母版（至少保留垂直+水平双族）",
    "narrow_passage": "narrow passage",
    "orientation_sensitive_passage": "orientation-sensitive passage",
    "wall_protrusion_bracket": "wall + protrusion / bracket",
    "frame_doorway": "frame / doorway",
    "pillar_wall": "pillar + wall",
    "staggered_corridor": "多障碍 staggered corridor",
    "mixed_industrial": "mixed industrial structure",
}

GUARANTEED_MULTI_ROUTE_FAMILIES = {
    "sparse_obb_clutter", "central_block", "multi_homotopy",
    "frame_doorway", "pillar_wall",
}
OPTIONAL_MULTI_ROUTE_FAMILIES = {
    "staggered_corridor", "mixed_industrial", "wall_protrusion_bracket",
}


def _endpoint_regions(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    # Keep the complete terminal regions inside SAMPLING_BOUNDS even after the
    # template's maximum global yaw/translation.  They are still genuine
    # regions (rather than fixed points), but no sampled certificate endpoint
    # can become invalid merely because the whole assembly was transformed.
    y_center = rng.uniform(-0.30, 0.30)
    y_size = rng.uniform(0.75, 0.95)
    # Both endpoints use broad, overlapping vertical regions.  Independent
    # rejection samples may therefore lie on nearly the same horizontal plane
    # or have a large height difference; the generator does not prescribe one
    # terminal as "high" and the other as "low".
    start_z = rng.uniform(1.82, 2.18)
    goal_z = rng.uniform(1.82, 2.18)
    start_z_size = min(
        rng.uniform(1.85, 2.18),
        2 * min(start_z - 0.72, 3.28 - start_z),
    )
    goal_z_size = min(
        rng.uniform(1.85, 2.18),
        2 * min(goal_z - 0.72, 3.28 - goal_z),
    )
    return (
        {
            "center": [-2.86, y_center, start_z],
            "size_xyz": [0.28, y_size, start_z_size],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "center": [2.86, -y_center, goal_z],
            "size_xyz": [0.28, y_size, goal_z_size],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    )


def _sample_region(region: dict[str, Any], rng: random.Random) -> list[float]:
    center = region["center"]
    size = region["size_xyz"]
    return [
        rng.uniform(center[index] - size[index] / 2, center[index] + size[index] / 2)
        for index in range(3)
    ]


def _a_star_route(
    start: list[float], goal: list[float], obstacles: list[dict[str, Any]],
    resolution: float = 0.28,
) -> list[list[float]] | None:
    lower = np.asarray([-3.25, -3.15, 0.78], dtype=np.float64)
    upper = np.asarray([3.25, 3.15, 3.32], dtype=np.float64)
    shape = tuple(int(math.floor((upper[i] - lower[i]) / resolution)) + 1 for i in range(3))

    def index_of(point: list[float]) -> tuple[int, int, int]:
        raw = np.rint((np.asarray(point) - lower) / resolution).astype(int)
        return tuple(int(np.clip(raw[i], 0, shape[i] - 1)) for i in range(3))

    def point_of(index: tuple[int, int, int]) -> list[float]:
        return (lower + resolution * np.asarray(index)).tolist()

    start_index, goal_index = index_of(start), index_of(goal)
    exact_start = [*start, 1.0, 0.0, 0.0, 0.0]
    exact_goal = [*goal, 1.0, 0.0, 0.0, 0.0]
    if not pose_is_free(exact_start, obstacles) or not pose_is_free(exact_goal, obstacles):
        return None
    free_cache: dict[tuple[int, int, int], bool] = {}

    def free(index: tuple[int, int, int]) -> bool:
        if index not in free_cache:
            free_cache[index] = pose_is_free(
                [*point_of(index), 1.0, 0.0, 0.0, 0.0], obstacles
            )
        return free_cache[index]

    if not free(start_index) or not free(goal_index):
        return None
    frontier: list[tuple[float, float, tuple[int, int, int]]] = []
    heapq.heappush(frontier, (0.0, 0.0, start_index))
    costs = {start_index: 0.0}
    parents: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    directions = (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    )
    found = False
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current == goal_index:
            found = True
            break
        if cost > costs.get(current, float("inf")):
            continue
        for direction in directions:
            neighbor = tuple(current[i] + direction[i] for i in range(3))
            if any(neighbor[i] < 0 or neighbor[i] >= shape[i] for i in range(3)):
                continue
            if not free(neighbor):
                continue
            next_cost = cost + resolution
            if next_cost >= costs.get(neighbor, float("inf")):
                continue
            costs[neighbor] = next_cost
            parents[neighbor] = current
            heuristic = resolution * sum(abs(neighbor[i] - goal_index[i]) for i in range(3))
            heapq.heappush(frontier, (next_cost + heuristic, next_cost, neighbor))
    if not found:
        return None
    indices = [goal_index]
    while indices[-1] != start_index:
        indices.append(parents[indices[-1]])
    indices.reverse()
    path = [exact_start]
    path.extend([[*point_of(index), 1.0, 0.0, 0.0, 0.0] for index in indices])
    path.append(exact_goal)
    if not route_is_free(path[:2], obstacles) or not route_is_free(path[-2:], obstacles):
        return None
    # Greedy line-of-sight simplification keeps the certificate compact.
    simplified = [path[0]]
    cursor = 0
    while cursor < len(path) - 1:
        target = len(path) - 1
        while target > cursor + 1 and not route_is_free(
            [path[cursor], path[target]], obstacles
        ):
            target -= 1
        simplified.append(path[target])
        cursor = target
    return simplified


def _find_upright_route(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
) -> list[list[float]] | None:
    for _ in range(18):
        start, goal = _sample_region(start_region, rng), _sample_region(goal_region, rng)
        route = _a_star_route(start, goal, obstacles)
        if route is not None:
            return route
    return None


def _route_through_side_corridor(
    start: list[float], goal: list[float], before: list[float],
    after: list[float], obstacles: list[dict[str, Any]], *,
    allow_astar: bool,
) -> list[list[float]] | None:
    """Connect shared terminals through one protected lateral corridor."""
    direct = [
        [*start, 1.0, 0.0, 0.0, 0.0],
        [*before, 1.0, 0.0, 0.0, 0.0],
        [*after, 1.0, 0.0, 0.0, 0.0],
        [*goal, 1.0, 0.0, 0.0, 0.0],
    ]
    if route_is_free(direct, obstacles):
        return direct
    if not allow_astar:
        return None
    if not route_is_free(direct[1:3], obstacles):
        return None
    approach = _a_star_route(start, before, obstacles)
    departure = _a_star_route(after, goal, obstacles)
    if approach is None or departure is None:
        return None
    candidate = [*approach, *direct[2:3], *departure[1:]]
    return candidate if route_is_free(candidate, obstacles) else None


def _diverse_side_routes(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
    separator: dict[str, Any] | None = None,
    *, allow_astar: bool = True, include_above: bool = False,
) -> list[tuple[str, list[list[float]]]] | None:
    """Find shared-endpoint lateral and optional vertical certificates.

    These are practical separator-crossing modes rather than a claim of a
    complete mathematical homotopy classification in SE(3).
    """
    robot_width = robot_projection_size(0.0)[0]
    if separator is None:
        center = np.asarray([0.0, 0.0, 1.8], dtype=np.float64)
        world_half = np.asarray([0.55, 0.55, 1.0], dtype=np.float64)
    else:
        center, rotation, half = _obstacle_obb(separator)
        world_half = np.abs(rotation) @ half
    x_clearance = float(world_half[0] + rng.uniform(0.65, 0.85))
    minimum_side = float(world_half[1] + robot_width / 2 + 0.18)
    minimum_side = max(1.30, minimum_side)
    if minimum_side > 2.48:
        return None
    for _ in range(12):
        start = _sample_region(start_region, rng)
        goal = _sample_region(goal_region, rng)
        crossing_z = min(2.65, max(1.15, 0.5 * (start[2] + goal[2])))
        routes: list[tuple[str, list[list[float]]]] = []
        for topology_id, sign in (("left", 1.0), ("right", -1.0)):
            route = None
            side_candidates = [
                min(2.48, minimum_side + offset)
                for offset in (rng.uniform(0.05, 0.18), 0.34, 0.62)
            ]
            for magnitude in side_candidates:
                side_y = float(center[1] + sign * magnitude)
                if abs(side_y) > 2.62:
                    continue
                before = [float(center[0] - x_clearance), side_y, crossing_z]
                after = [float(center[0] + x_clearance), side_y, crossing_z]
                route = _route_through_side_corridor(
                    start, goal, before, after, obstacles,
                    allow_astar=allow_astar,
                )
                if route is not None:
                    break
            if route is None:
                break
            routes.append((topology_id, route))
        if len(routes) == 2:
            if include_above and separator is not None:
                robot_height = robot_projection_size(0.0)[1]
                above_z = float(
                    center[2] + world_half[2]
                    + robot_height / 2 + ROBOT.safety_margin + 0.10
                )
                if above_z <= 3.28:
                    above = _route_through_side_corridor(
                        start, goal,
                        [float(center[0] - x_clearance), float(center[1]), above_z],
                        [float(center[0] + x_clearance), float(center[1]), above_z],
                        obstacles, allow_astar=allow_astar,
                    )
                    if above is not None:
                        routes.append(("above", above))
            return routes
    return None


def _frame_route_modes(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
) -> list[tuple[str, list[list[float]]]] | None:
    """Preserve through/above/lateral practical modes around a doorway."""
    portal = next((item for item in obstacles if "aperture_center_y" in item), None)
    if portal is None:
        return None
    wall_x = float(portal.get("wall_x", 0.0))
    center_y = float(portal["aperture_center_y"])
    center_z = float(portal["aperture_center_z"])
    frame_outer_half = 0.5 * (
        float(portal["aperture_width"])
        + 2 * max(float(item["size_xyz"][1]) for item in obstacles[:2])
    )
    robot_width = robot_projection_size(0.0)[0]
    bypass_offset = frame_outer_half + robot_width / 2 + 0.18
    # The workspace extends to +/-4 m.  Keep enough room for the full vehicle
    # envelope, but do not reject a valid outer bypass merely because its
    # centre is a few centimetres beyond the old visualization heuristic.
    if max(abs(center_y + bypass_offset), abs(center_y - bypass_offset)) > 2.72:
        return None
    quaternion = yaw_quaternion(math.pi / 2)
    rotated_center = quaternion_matrix(quaternion) @ np.asarray(ROBOT.center)
    through_y = center_y - float(rotated_center[1])
    for _ in range(35):
        start = _sample_region(start_region, rng)
        goal = _sample_region(goal_region, rng)
        z = min(2.65, max(1.05, center_z + rng.uniform(-0.12, 0.12)))
        through = [
            [*start, *quaternion],
            [wall_x - 1.35, through_y, z, *quaternion],
            [wall_x + 1.35, through_y, z, *quaternion],
            [*goal, *quaternion],
        ]
        routes: list[tuple[str, list[list[float]]]] = [("through", through)]
        header = next((item for item in obstacles if item["id"] == "frame_header"), None)
        if header is not None:
            header_center, header_rotation, header_half = _obstacle_obb(header)
            header_world_half = np.abs(header_rotation) @ header_half
            over_z = float(
                header_center[2] + header_world_half[2]
                + robot_projection_size(0.0)[1] / 2 + 0.12
            )
            if over_z <= 3.28:
                above = [
                    [*start, *quaternion],
                    [wall_x - 1.35, center_y, over_z, *quaternion],
                    [wall_x + 1.35, center_y, over_z, *quaternion],
                    [*goal, *quaternion],
                ]
                if route_is_free(above, obstacles):
                    routes.append(("above", above))
        for topology_id, side_y in (
            ("left", center_y + bypass_offset),
            ("right", center_y - bypass_offset),
        ):
            bypass = [
                [*start, *quaternion],
                [wall_x - 1.35, side_y, z, *quaternion],
                [wall_x + 1.35, side_y, z, *quaternion],
                [*goal, *quaternion],
            ]
            if route_is_free(bypass, obstacles):
                routes.append((topology_id, bypass))
        if route_is_free(through, obstacles) and len(routes) >= 2:
            return routes
    return None


def _multi_homotopy_routes(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
) -> list[tuple[str, list[list[float]]]] | None:
    separator = next(
        (item for item in obstacles if item.get("homotopy_separator")), None
    )
    if separator is None:
        return None
    center, rotation, half = _obstacle_obb(separator)
    world_half = np.abs(rotation) @ half
    robot_width, robot_height = robot_projection_size(0.0)
    clearance_y = float(
        world_half[1] + robot_width / 2 + rng.uniform(0.14, 0.22)
    )
    side_values = {
        "left": float(center[1] + clearance_y),
        "right": float(center[1] - clearance_y),
    }
    if any(abs(value) > 2.72 for value in side_values.values()):
        return None
    above_z = float(center[2] + world_half[2] + robot_height / 2 + 0.12)
    below_z = float(center[2] - world_half[2] - robot_height / 2 - 0.12)
    if below_z < 0.84 or above_z > 3.28:
        return None
    x_clearance = float(world_half[0] + rng.uniform(0.78, 0.96))
    for _ in range(35):
        start, goal = _sample_region(start_region, rng), _sample_region(goal_region, rng)
        routes: list[tuple[str, list[list[float]]]] = []
        specifications = (
            ("above", float(center[1]), above_z),
            ("below", float(center[1]), below_z),
            ("left", side_values["left"], float(center[2])),
            ("right", side_values["right"], float(center[2])),
        )
        for topology_id, crossing_y, crossing_z in specifications:
            route = _route_through_side_corridor(
                start, goal,
                [float(center[0] - x_clearance), crossing_y, crossing_z],
                [float(center[0] + x_clearance), crossing_y, crossing_z],
                obstacles, allow_astar=False,
            )
            if route is None:
                break
            routes.append((topology_id, route))
        if len(routes) == len(specifications):
            return routes
    return None


def _randomize_terminal_attitudes(
    routes: list[tuple[str, list[list[float]]]],
    obstacles: list[dict[str, Any]], rng: random.Random,
) -> list[tuple[str, list[list[float]]]]:
    """Sample non-level endpoints inside the configured flight envelope."""
    # Shared endpoints of dense industrial/bracket candidates have less free
    # angular clearance than open separator scenes.  A moderate distribution
    # preserves meaningful attitude variation without spending hundreds of
    # rejection tests on near-limit terminal poses.
    compact_multi_candidate = any(
        topology_id.startswith(("winding", "reverse_winding", "dogleg"))
        for topology_id, _ in routes
    )
    roll_upper = 18.0 if compact_multi_candidate else MAX_FLIGHT_ROLL_DEG
    pitch_upper = 30.0 if compact_multi_candidate else MAX_FLIGHT_PITCH_DEG
    for _ in range(100):
        endpoint_quaternions = []
        for base_pose in (routes[0][1][0], routes[0][1][-1]):
            roll = math.radians(
                rng.uniform(4.0, roll_upper) * rng.choice((-1, 1))
            )
            pitch = math.radians(
                rng.uniform(3.0, pitch_upper) * rng.choice((-1, 1))
            )
            yaw = math.radians(rng.uniform(-18.0, 18.0))
            endpoint_quaternions.append(quaternion_multiply(
                base_pose[3:7], rpy_quaternion(roll, pitch, yaw),
            ))
        candidates = []
        for topology_id, route in routes:
            candidate = [pose.copy() for pose in route]
            candidate[0][3:7] = endpoint_quaternions[0]
            candidate[-1][3:7] = endpoint_quaternions[1]
            for index in range(1, len(candidate) - 1):
                base_quaternion = candidate[index][3:7]
                base_rotation = quaternion_matrix(base_quaternion)
                base_tilt = math.degrees(math.acos(
                    min(1.0, max(-1.0, float(base_rotation[2, 2])))
                ))
                # Preserve the deliberately rolled gate attitude.  Other
                # guide poses receive mild flight attitude so the expert does
                # not collapse to an all-level quaternion curve.
                if base_tilt > 15.0:
                    continue
                if topology_id == "left":
                    roll_sign = 1
                elif topology_id == "right":
                    roll_sign = -1
                else:
                    roll_sign = rng.choice((-1, 1))
                delta = rpy_quaternion(
                    math.radians(roll_sign * rng.uniform(4.0, 10.0)),
                    math.radians(rng.uniform(2.5, 7.0) * rng.choice((-1, 1))),
                    math.radians(rng.uniform(-7.0, 7.0)),
                )
                candidate[index][3:7] = quaternion_multiply(
                    base_quaternion, delta,
                )
            candidates.append((topology_id, candidate))
        if all(
            route_is_free(route, obstacles)
            and all(
                attitude_is_within_flight_limits(pose[3:7])
                for pose in _interpolate_route(route)
            )
            for _, route in candidates
        ):
            return candidates
    # Tight scenes may not tolerate attitude changes at every A* corner.
    # Retain guaranteed non-level endpoints with a smaller envelope instead of
    # silently reverting to identity attitudes.
    for _ in range(80):
        endpoint_quaternions = []
        for base_pose in (routes[0][1][0], routes[0][1][-1]):
            delta = rpy_quaternion(
                math.radians(rng.uniform(3.0, 6.0) * rng.choice((-1, 1))),
                math.radians(rng.uniform(2.0, 4.0) * rng.choice((-1, 1))),
                math.radians(rng.uniform(-4.0, 4.0)),
            )
            endpoint_quaternions.append(quaternion_multiply(base_pose[3:7], delta))
        candidates = []
        for topology_id, route in routes:
            candidate = [pose.copy() for pose in route]
            candidate[0][3:7] = endpoint_quaternions[0]
            candidate[-1][3:7] = endpoint_quaternions[1]
            candidates.append((topology_id, candidate))
        if all(
            route_is_free(route, obstacles)
            and all(
                attitude_is_within_flight_limits(pose[3:7])
                for pose in _interpolate_route(route)
            )
            for _, route in candidates
        ):
            return candidates
    # A deterministic sign search avoids rejecting an otherwise valid scene
    # when a terminal happens to sit close to one side of an obstacle.  The
    # magnitude remains visibly non-level and inside the same hard limits.
    small_deltas = [
        rpy_quaternion(
            math.radians(roll_sign * roll),
            math.radians(pitch_sign * pitch), math.radians(yaw),
        )
        for roll in (3.0, 4.0, 5.0)
        for pitch in (2.0, 3.0)
        for yaw in (-4.0, -2.0, 0.0, 2.0, 4.0)
        for roll_sign in (-1, 1)
        for pitch_sign in (-1, 1)
    ]
    rng.shuffle(small_deltas)
    delta_pairs = [(delta, delta) for delta in small_deltas]
    delta_pairs.extend(
        (rng.choice(small_deltas), rng.choice(small_deltas))
        for _ in range(1200)
    )
    for start_delta, goal_delta in delta_pairs:
        endpoint_quaternions = [
            quaternion_multiply(routes[0][1][0][3:7], start_delta),
            quaternion_multiply(routes[0][1][-1][3:7], goal_delta),
        ]
        candidates = []
        for topology_id, route in routes:
            candidate = [pose.copy() for pose in route]
            candidate[0][3:7] = endpoint_quaternions[0]
            candidate[-1][3:7] = endpoint_quaternions[1]
            candidates.append((topology_id, candidate))
        if all(
            route_is_free(route, obstacles)
            and all(
                attitude_is_within_flight_limits(pose[3:7])
                for pose in _interpolate_route(route)
            )
            for _, route in candidates
        ):
            return candidates
    raise ValueError("could not sample collision-free non-level terminal attitudes")


def _portal_route(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
) -> list[list[float]] | None:
    portal = next(
        (item for item in obstacles if "aperture_center_y" in item), None
    )
    if portal is None:
        return None
    wall_x = float(portal.get("wall_x", portal["pose"]["position"][0]))
    center_y = float(portal["aperture_center_y"])
    center_z = float(portal.get("aperture_center_z", rng.uniform(1.15, 2.35)))
    quaternion = yaw_quaternion(math.pi / 2)
    rotated_center = quaternion_matrix(quaternion) @ np.asarray(ROBOT.center)
    reference_y = center_y - float(rotated_center[1])
    for _ in range(25):
        start, goal = _sample_region(start_region, rng), _sample_region(goal_region, rng)
        z = min(2.8, max(0.95, center_z + rng.uniform(-0.18, 0.18)))
        route = [
            [*start, *quaternion],
            [wall_x - 1.35, reference_y, z, *quaternion],
            [wall_x + 1.35, reference_y, z, *quaternion],
            [*goal, *quaternion],
        ]
        if route_is_free(route, obstacles):
            return route
    return None


def _bracket_dogleg_routes(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
) -> list[tuple[str, list[list[float]]]] | None:
    """Certify both lateral doglegs around the bracket tip when feasible."""
    portal = next(
        (item for item in obstacles if item["id"].startswith("bracket_wall")),
        None,
    )
    tip = next((item for item in obstacles if item["id"] == "bracket_tip"), None)
    if portal is None or tip is None:
        return None
    wall_x = float(portal["wall_x"])
    gap_center = float(portal["aperture_center_y"])
    tip_x = float(tip["pose"]["position"][0])
    quaternion = yaw_quaternion(math.pi / 2)
    rotated_center = quaternion_matrix(quaternion) @ np.asarray(ROBOT.center)
    portal_reference_y = gap_center - float(rotated_center[1])
    tip_sign = 1.0 if gap_center >= 0.0 else -1.0
    robot_width = min(ROBOT.size[0], ROBOT.size[1]) + 2 * ROBOT.safety_margin
    detour_distance = (
        float(tip["size_xyz"][1]) / 2 + robot_width / 2 + 0.18
    )
    # Move toward the workspace centre before returning to the offset portal.
    best_routes: list[tuple[str, list[list[float]]]] = []
    for _ in range(60):
        start = _sample_region(start_region, rng)
        goal = _sample_region(goal_region, rng)
        z = rng.uniform(1.48, 2.08)
        routes = []
        for topology_id, direction in (
            ("dogleg_inner", -tip_sign),
            ("dogleg_outer", tip_sign),
        ):
            detour_y = portal_reference_y + direction * detour_distance
            route = [
                [*start, *quaternion],
                [tip_x - 0.62, detour_y, z, *quaternion],
                [tip_x + 1.04, detour_y, z, *quaternion],
                [tip_x + 1.04, portal_reference_y, z, *quaternion],
                [wall_x - 0.34, portal_reference_y, z, *quaternion],
                [wall_x + 0.58, portal_reference_y, z, *quaternion],
                [*goal, *quaternion],
            ]
            if route_is_free(route, obstacles):
                routes.append((topology_id, route))
        if len(routes) > len(best_routes):
            best_routes = routes
        if len(routes) == 2:
            return routes
    return best_routes or None


def _mixed_industrial_routes(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
) -> list[tuple[str, list[list[float]]]] | None:
    """Certify under-beam and over-beam industrial route candidates."""
    machine = next(
        (item for item in obstacles if item["id"] == "machine_tower"), None
    )
    control = next(
        (item for item in obstacles if item["id"] == "control_box"), None
    )
    beam = next(
        (item for item in obstacles if item["id"] == "supported_pipe"), None
    )
    if machine is None or control is None or beam is None:
        return None
    machine_x, machine_y, _ = machine["pose"]["position"]
    control_x, control_y, _ = control["pose"]["position"]
    beam_x = float(beam["pose"]["position"][0])
    beam_bottom = (
        float(beam["pose"]["position"][2])
        - float(beam["size_xyz"][2]) / 2
    )
    robot_height = robot_projection_size(0.0)[1]
    under_z = beam_bottom - robot_height / 2 - 0.10
    beam_top = (
        float(beam["pose"]["position"][2])
        + float(beam["size_xyz"][2]) / 2
    )
    over_z = beam_top + robot_height / 2 + ROBOT.safety_margin + 0.08
    quaternion = [1.0, 0.0, 0.0, 0.0]
    configurations = [
        (first_magnitude, second_magnitude, z_delta)
        for first_magnitude in (1.00, 1.10, 1.20, 1.30, 1.40, 1.48)
        for second_magnitude in (1.00, 1.10, 1.20, 1.30, 1.40)
        for z_delta in (-0.20, -0.16, -0.10, -0.04, 0.04)
    ]
    rng.shuffle(configurations)
    for _ in range(30):
        start = _sample_region(start_region, rng)
        goal = _sample_region(goal_region, rng)
        for first_magnitude, second_magnitude, z_delta in configurations:
            first_y = -math.copysign(first_magnitude, machine_y)
            second_y = -math.copysign(second_magnitude, control_y)
            z = min(1.72, max(1.10, under_z + z_delta))
            winding_under = [
                [*start, *quaternion],
                [float(machine_x - 0.95), first_y, z, *quaternion],
                [beam_x - 0.12, first_y, z, *quaternion],
                [beam_x + 0.12, second_y, z, *quaternion],
                [float(control_x + 1.25), second_y, z, *quaternion],
                [*goal, *quaternion],
            ]
            if not route_is_free(winding_under, obstacles):
                continue
            routes = [("winding_under", winding_under)]
            reverse_first_y = math.copysign(2.08, machine_y)
            reverse_second_y = math.copysign(2.08, control_y)
            reverse_under = [
                [*start, *quaternion],
                [float(machine_x - 0.95), reverse_first_y, z, *quaternion],
                [beam_x - 0.72, reverse_first_y, z, *quaternion],
                [beam_x + 0.72, reverse_second_y, z, *quaternion],
                [float(control_x + 1.25), reverse_second_y, z, *quaternion],
                [*goal, *quaternion],
            ]
            if route_is_free(reverse_under, obstacles):
                routes.append(("reverse_winding_under", reverse_under))
            if over_z <= SAMPLING_BOUNDS["max"][2] - 0.12:
                winding_over = [
                    [*start, *quaternion],
                    [float(machine_x - 0.95), first_y, z, *quaternion],
                    [beam_x - 0.72, first_y, over_z, *quaternion],
                    [beam_x + 0.72, second_y, over_z, *quaternion],
                    [float(control_x + 1.25), second_y, z, *quaternion],
                    [*goal, *quaternion],
                ]
                if route_is_free(winding_over, obstacles):
                    routes.append(("winding_over", winding_over))
            return routes
    return None


def _orientation_route(
    obstacles: list[dict[str, Any]], start_region: dict[str, Any],
    goal_region: dict[str, Any], rng: random.Random,
) -> list[list[float]] | None:
    metadata = next((item for item in obstacles if "required_roll_deg" in item), None)
    if metadata is None:
        return None
    aperture = np.asarray(metadata["aperture_center_xyz"], dtype=np.float64)
    required = float(metadata["required_roll_deg"])
    alternatives = list(np.linspace(
        max(-MAX_FLIGHT_ROLL_DEG, required - 2.0),
        min(MAX_FLIGHT_ROLL_DEG, required + 2.0), 17,
    ))
    rng.shuffle(alternatives)
    candidates = [required, *alternatives]
    for roll_degrees in candidates:
        start, goal = _sample_region(start_region, rng), _sample_region(goal_region, rng)
        rolled = rpy_quaternion(math.radians(float(roll_degrees)), 0.0, 0.0)
        crossing = aperture - quaternion_matrix(rolled) @ np.asarray(ROBOT.center)
        route = [
            [*start, 1.0, 0.0, 0.0, 0.0],
            [crossing[0] - 2.25, crossing[1], crossing[2], 1.0, 0.0, 0.0, 0.0],
            [crossing[0] - 1.75, crossing[1], crossing[2], *rolled],
            [crossing[0] - 0.45, crossing[1], crossing[2], *rolled],
            [crossing[0] + 0.45, crossing[1], crossing[2], *rolled],
            [crossing[0] + 1.75, crossing[1], crossing[2], *rolled],
            [crossing[0] + 2.25, crossing[1], crossing[2], 1.0, 0.0, 0.0, 0.0],
            [*goal, 1.0, 0.0, 0.0, 0.0],
        ]
        if route_is_free(route, obstacles):
            return route
    return None


def _transform_obstacle(obstacle: dict[str, Any], p: SceneParameters) -> None:
    angle = math.radians(p.global_yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = obstacle["pose"]["position"]
    if "aperture_center_y" in obstacle:
        portal_x = float(obstacle.get("wall_x", x))
        portal_y = float(obstacle["aperture_center_y"])
        obstacle["aperture_center_world_xy"] = [
            round(cosine * portal_x - sine * portal_y + p.translate_x, 6),
            round(sine * portal_x + cosine * portal_y + p.translate_y, 6),
        ]
    if "aperture_center_xyz" in obstacle:
        px, py, pz = obstacle["aperture_center_xyz"]
        obstacle["aperture_center_world_xyz"] = [
            round(cosine * px - sine * py + p.translate_x, 6),
            round(sine * px + cosine * py + p.translate_y, 6),
            round(float(pz), 6),
        ]
    obstacle["pose"]["position"] = [
        round(cosine * x - sine * y + p.translate_x, 6),
        round(sine * x + cosine * y + p.translate_y, 6),
        z,
    ]
    gw, gx, gy, gz = yaw_quaternion(angle)
    lw, lx, ly, lz = obstacle["pose"]["quaternion_wxyz"]
    obstacle["pose"]["quaternion_wxyz"] = [round(value, 9) for value in (
        gw * lw - gx * lx - gy * ly - gz * lz,
        gw * lx + gx * lw + gy * lz - gz * ly,
        gw * ly - gx * lz + gy * lw + gz * lx,
        gw * lz + gx * ly - gy * lx + gz * lw,
    )]


def _transform_pose(pose: list[float], p: SceneParameters) -> list[float]:
    angle = math.radians(p.global_yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = pose[:3]
    global_q = yaw_quaternion(angle)
    local_q = pose[3:7]
    gw, gx, gy, gz = global_q
    lw, lx, ly, lz = local_q
    quaternion = [
        gw * lw - gx * lx - gy * ly - gz * lz,
        gw * lx + gx * lw + gy * lz - gz * ly,
        gw * ly - gx * lz + gy * lw + gz * lx,
        gw * lz + gx * ly - gy * lx + gz * lw,
    ]
    return [
        round(cosine * x - sine * y + p.translate_x, 6),
        round(sine * x + cosine * y + p.translate_y, 6),
        z, *[round(float(value), 9) for value in quaternion],
    ]


def _transform_region(region: dict[str, Any], p: SceneParameters) -> dict[str, Any]:
    pose = _transform_pose([
        *region["center"], *region.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
    ], p)
    return {
        "center": pose[:3], "size_xyz": region["size_xyz"],
        "quaternion_wxyz": pose[3:7],
    }


def _route_pose_at_fraction(
    route: list[list[float]], fraction: float,
) -> list[float]:
    """Interpolate one SE(3) pose by translational arc-length fraction."""
    states = np.asarray(route, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = float(np.clip(fraction, 0.0, 1.0)) * cumulative[-1]
    segment = min(
        max(int(np.searchsorted(cumulative, target, side="right") - 1), 0),
        len(states) - 2,
    )
    alpha = (target - cumulative[segment]) / max(lengths[segment], 1e-9)
    position = (1.0 - alpha) * states[segment, :3] + alpha * states[segment + 1, :3]
    first_q = states[segment, 3:7].copy()
    second_q = states[segment + 1, 3:7].copy()
    if float(first_q @ second_q) < 0.0:
        second_q *= -1.0
    quaternion = (1.0 - alpha) * first_q + alpha * second_q
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return [*position.tolist(), *quaternion.tolist()]


def _route_sampling_frame(
    route: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Return travel/lateral/up axes and an oriented-region quaternion."""
    travel = np.asarray(route[-1][:2]) - np.asarray(route[0][:2])
    travel /= max(float(np.linalg.norm(travel)), 1e-9)
    travel3 = np.asarray([*travel, 0.0], dtype=np.float64)
    lateral3 = np.asarray([-travel[1], travel[0], 0.0], dtype=np.float64)
    up3 = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    region_quaternion = yaw_quaternion(
        math.atan2(float(travel[1]), float(travel[0]))
    )
    return travel3, lateral3, up3, region_quaternion


def _robot_projection_relative_to_reference(
    pose: list[float], axis: np.ndarray,
) -> tuple[float, float]:
    """Project the safety-inflated URDF primitives relative to base_link."""
    reference = np.asarray(pose[:3], dtype=np.float64)
    low, high = float("inf"), -float("inf")
    for center, rotation, half in _robot_primitive_obbs(pose):
        radius = float(np.sum(np.abs(axis @ rotation) * half))
        relative = float((center - reference) @ axis)
        low = min(low, relative - radius)
        high = max(high, relative + radius)
    return low, high


def _portal_sampling_regions(
    family: str, topology_id: str, route: list[list[float]],
    obstacles: list[dict[str, Any]],
    reference_poses: tuple[list[float], list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Create two coherent cross-sections that must connect through a portal."""
    portal = next(
        (item for item in obstacles if "aperture_width" in item), None
    )
    if portal is None:
        return []
    travel3, lateral3, _, region_quaternion = _route_sampling_frame(route)
    aperture_xy = portal.get("aperture_center_world_xy")
    if aperture_xy is None:
        aperture_xy = [
            float(portal.get("wall_x", portal["pose"]["position"][0])),
            float(portal["aperture_center_y"]),
        ]
    aperture_center = np.asarray([*aperture_xy, 0.0], dtype=np.float64)
    aperture_lateral = float(aperture_center @ lateral3)
    gap_half = float(portal["aperture_width"]) / 2
    mode_constraint = {
        "frame_doorway": "doorway_aperture_cross_section",
        "narrow_passage": "narrow_aperture_cross_section",
        "wall_protrusion_bracket": "bracket_aperture_cross_section",
    }[family]
    regions = []
    # The certified portal routes deliberately store one pose on either side
    # of the wall.  Keeping their lateral/Z samples coherent makes the segment
    # between them cross the actual opening instead of wandering around it.
    portal_references = reference_poses or (route[1], route[-2])
    for index, reference_pose in enumerate(portal_references):
        center_pose = list(reference_pose)
        lateral_low, lateral_high = _robot_projection_relative_to_reference(
            center_pose, lateral3,
        )
        allowed_lateral_low = (
            aperture_lateral - gap_half - lateral_low + 0.025
        )
        allowed_lateral_high = (
            aperture_lateral + gap_half - lateral_high - 0.025
        )
        reference_lateral = float(np.asarray(center_pose[:3]) @ lateral3)
        if allowed_lateral_high <= allowed_lateral_low + 0.025:
            allowed_lateral_low = reference_lateral - 0.025
            allowed_lateral_high = reference_lateral + 0.025

        z_low = SAMPLING_BOUNDS["min"][2] + 0.10
        z_high = SAMPLING_BOUNDS["max"][2] - 0.10
        if family == "frame_doorway":
            clear_height = 2 * float(portal["aperture_center_z"])
            robot_z_low, robot_z_high = _robot_projection_relative_to_reference(
                center_pose, np.asarray([0.0, 0.0, 1.0]),
            )
            z_low = max(z_low, 0.06 - robot_z_low)
            z_high = min(z_high, clear_height - 0.06 - robot_z_high)
        if z_high <= z_low + 0.05:
            z_low = float(reference_pose[2]) - 0.025
            z_high = float(reference_pose[2]) + 0.025

        lateral_center = 0.5 * (allowed_lateral_low + allowed_lateral_high)
        delta_lateral = lateral_center - reference_lateral
        center_pose[0] += delta_lateral * lateral3[0]
        center_pose[1] += delta_lateral * lateral3[1]
        center_pose[2] = 0.5 * (z_low + z_high)
        regions.append({
            "region_id": f"{topology_id}_portal_{index}",
            "center_pose": center_pose,
            "size_xyz": [
                0.52,
                round(float(allowed_lateral_high - allowed_lateral_low), 6),
                round(float(z_high - z_low), 6),
            ],
            "quaternion_wxyz": region_quaternion,
            "orientation_sampling": "bounded_reference_rpy_jitter",
            "orientation_rpy_jitter_deg": [1.5, 1.5, 3.0],
            "mode_constraint": mode_constraint,
            "coherent_local_axes": [1, 2],
            "coherent_sampling_strength": [0.0, 1.0, 1.0],
        })
    return regions


def _free_axis_extent(
    pose: list[float], axis: np.ndarray, obstacles: list[dict[str, Any]],
    maximum: float,
) -> tuple[float, float]:
    """Find a contiguous collision-free displacement interval on one axis."""
    reference = np.asarray(pose[:3], dtype=np.float64)

    def free(offset: float) -> bool:
        position = reference + offset * axis
        if any(
            position[index] < SAMPLING_BOUNDS["min"][index] + 0.10
            or position[index] > SAMPLING_BOUNDS["max"][index] - 0.10
            for index in range(3)
        ):
            return False
        candidate = [*position.tolist(), *pose[3:7]]
        return pose_is_free(candidate, obstacles)

    extents = []
    for sign in (-1.0, 1.0):
        last_free = 0.0
        probe = 0.08
        while probe <= maximum + 1e-9 and free(sign * probe):
            last_free = probe
            probe += 0.08
        low, high = last_free, min(maximum, probe)
        for _ in range(8):
            middle = 0.5 * (low + high)
            if free(sign * middle):
                low = middle
            else:
                high = middle
        extents.append(low)
    return extents[0], extents[1]


def _bracket_dogleg_sampling_regions(
    topology_id: str, route: list[list[float]],
    obstacles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose the bracket detour and wall aperture as one ordered guide chain."""
    travel3, lateral3, up3, region_quaternion = _route_sampling_frame(route)
    regions = []
    for index, reference_pose in enumerate(route[1:3]):
        lateral_negative, lateral_positive = _free_axis_extent(
            reference_pose, lateral3, obstacles, 0.24,
        )
        z_negative, z_positive = _free_axis_extent(
            reference_pose, up3, obstacles, 0.25,
        )
        lateral_half = max(0.08, min(lateral_negative, lateral_positive))
        z_half = max(0.08, min(z_negative, z_positive))
        regions.append({
            "region_id": f"{topology_id}_dogleg_{index}",
            "center_pose": list(reference_pose),
            "size_xyz": [0.46, 2 * lateral_half, 2 * z_half],
            "quaternion_wxyz": region_quaternion,
            "orientation_sampling": "bounded_reference_rpy_jitter",
            "orientation_rpy_jitter_deg": [2.5, 2.5, 4.0],
            "mode_constraint": "bracket_dogleg_clearance",
            "coherent_local_axes": [1, 2],
            "coherent_sampling_strength": [0.0, 0.76, 0.84],
        })
    portal_regions = _portal_sampling_regions(
        "wall_protrusion_bracket", topology_id, route, obstacles,
        reference_poses=(route[3], route[-2]),
    )
    for index, region in enumerate(portal_regions):
        region["region_id"] = f"{topology_id}_wall_portal_{index}"
        reference_z = float(route[3][2])
        region["center_pose"][2] = reference_z
        region["size_xyz"][2] = 0.52
    return [*regions, *portal_regions]


def _industrial_winding_sampling_regions(
    topology_id: str, route: list[list[float]],
    obstacles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose up to four broad stations from one industrial candidate."""
    _, lateral3, up3, region_quaternion = _route_sampling_frame(route)
    regions = []
    interior = route[1:-1]
    if len(interior) > 4:
        selected = sorted(set(
            int(round(value))
            for value in np.linspace(0, len(interior) - 1, 4)
        ))
        interior = [interior[index] for index in selected]
    for index, reference_pose in enumerate(interior):
        lateral_negative, lateral_positive = _free_axis_extent(
            reference_pose, lateral3, obstacles, 0.28,
        )
        z_negative, z_positive = _free_axis_extent(
            reference_pose, up3, obstacles, 0.24,
        )
        lateral_half = max(0.07, min(lateral_negative, lateral_positive))
        z_half = max(0.07, min(z_negative, z_positive))
        regions.append({
            "region_id": f"{topology_id}_industrial_station_{index}",
            "center_pose": list(reference_pose),
            "size_xyz": [0.42, 2 * lateral_half, 2 * z_half],
            "quaternion_wxyz": region_quaternion,
            "orientation_sampling": "bounded_reference_rpy_jitter",
            "orientation_rpy_jitter_deg": [1.5, 1.5, 3.0],
            "mode_constraint": "industrial_candidate_soft_channel",
            "coherent_local_axes": [1, 2],
            "coherent_sampling_strength": [0.0, 0.84, 1.0],
        })
    return regions


def _structure_aware_clearance_regions(
    family: str, topology_id: str, route: list[list[float]],
    obstacles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a clearance-adaptive chain for staggered/industrial layouts."""
    travel3, lateral3, up3, region_quaternion = _route_sampling_frame(route)
    start = np.asarray(route[0][:3], dtype=np.float64)
    goal = np.asarray(route[-1][:3], dtype=np.float64)
    longitudinal_length = float((goal - start) @ travel3)
    core = [
        item for item in obstacles
        if item.get("role") != "secondary_obstacle"
        and item.get("role") != "structural_support"
    ]
    fractions = []
    for obstacle in core:
        center = np.asarray(obstacle["pose"]["position"], dtype=np.float64)
        fraction = float((center - start) @ travel3) / max(
            longitudinal_length, 1e-9,
        )
        if 0.16 <= fraction <= 0.84:
            fractions.append(fraction)
    fractions.sort()
    stations: list[float] = []
    for fraction in fractions:
        if not stations or fraction - stations[-1] >= 0.12:
            stations.append(fraction)
    if len(stations) < 2:
        stations = [0.36, 0.64]
    if len(stations) > 4:
        indices = np.linspace(0, len(stations) - 1, 4).round().astype(int)
        stations = [stations[index] for index in indices]

    structure_center = (
        np.mean([
            np.asarray(item["pose"]["position"], dtype=np.float64)
            for item in core
        ], axis=0)
        if core else 0.5 * (start + goal)
    )
    mode_sign = 1.0 if topology_id == "left" else -1.0
    regions = []
    for index, fraction in enumerate(stations):
        reference_pose = _route_pose_at_fraction(route, fraction)
        clearance_lateral_limit = (
            0.58 if family == "mixed_industrial" else 1.15
        )
        clearance_z_limit = (
            0.48 if family == "mixed_industrial" else 0.85
        )
        negative, positive = _free_axis_extent(
            reference_pose, lateral3, obstacles, clearance_lateral_limit,
        )
        interval_low, interval_high = -negative, positive
        local_lateral = float(
            (np.asarray(reference_pose[:3]) - structure_center) @ lateral3
        )
        if topology_id in {"left", "right"}:
            mode_floor = max(0.18, 0.30 * abs(local_lateral))
            if mode_sign > 0:
                interval_low = max(interval_low, mode_floor - local_lateral)
            else:
                interval_high = min(interval_high, -mode_floor - local_lateral)
        if interval_high <= interval_low + 0.12:
            interval_low, interval_high = -0.06, 0.06
        lateral_shift = 0.5 * (interval_low + interval_high)
        center_pose = list(reference_pose)
        center_pose[0] += lateral_shift * lateral3[0]
        center_pose[1] += lateral_shift * lateral3[1]
        z_negative, z_positive = _free_axis_extent(
            center_pose, up3, obstacles, clearance_z_limit,
        )
        z_low, z_high = -z_negative, z_positive
        if z_high <= z_low + 0.16:
            z_low, z_high = -0.08, 0.08
        center_pose[2] += 0.5 * (z_low + z_high)
        regions.append({
            "region_id": f"{topology_id}_clearance_{index}",
            "center_pose": center_pose,
            "size_xyz": [
                0.58,
                round(float(interval_high - interval_low), 6),
                round(float(z_high - z_low), 6),
            ],
            "quaternion_wxyz": region_quaternion,
            "orientation_sampling": "bounded_reference_rpy_jitter",
            "orientation_rpy_jitter_deg": (
                [2.5, 2.5, 5.0]
                if family == "mixed_industrial" else [4.0, 4.0, 7.0]
            ),
            "mode_constraint": (
                "staggered_clearance_chain"
                if family == "staggered_corridor"
                else "industrial_clearance_chain"
            ),
            "coherent_local_axes": [1, 2],
            # Preserve a smooth family-level trend while still allowing each
            # obstacle station to move locally inside its own free cell.
            "coherent_sampling_strength": [0.0, 0.68, 0.82],
        })
    return regions


def _expert_planning_guides(
    family: str,
    topology_routes: list[tuple[str, list[list[float]]]],
    obstacles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize route guidance as part of the generated scene contract.

    Ordinary route modes expose loose proposal regions.  A planner may use a
    small subset as soft anchors, but the final smoother only constrains the
    task endpoints.  Thirty percent of attempts remain completely global.
    The orientation gate is deliberately strict: its aperture poses stay hard
    because pose and position feasibility are coupled there.
    """
    guides: list[dict[str, Any]] = []
    separator = next((
        obstacle for obstacle in obstacles
        if obstacle.get("route_separator")
        or obstacle.get("homotopy_separator")
    ), None)
    for topology_id, route in topology_routes:
        if family == "orientation_sensitive_passage":
            tilted = [
                pose for pose in route[1:-1]
                if max(abs(value) for value in quaternion_roll_pitch_degrees(pose[3:7])) > 15.0
            ]
            if len(tilted) < 2:
                raise ValueError(
                    "orientation-sensitive route has no fixed aperture poses"
                )
            fixed_waypoints = [route[0], tilted[0], tilted[-1], route[-1]]
            regions: list[dict[str, Any]] = []
            policy = "fixed_waypoints_required"
            strict_constraint = "orientation_coupled_aperture"
        else:
            fixed_waypoints = route
            strict_constraint = None
            if (
                family == "narrow_passage"
                or (family == "frame_doorway" and topology_id == "through")
            ):
                regions = _portal_sampling_regions(
                    family, topology_id, route, obstacles,
                )
            elif family == "wall_protrusion_bracket":
                regions = _bracket_dogleg_sampling_regions(
                    topology_id, route, obstacles,
                )
            elif family == "mixed_industrial":
                regions = _industrial_winding_sampling_regions(
                    topology_id, route, obstacles,
                )
            elif family == "staggered_corridor":
                regions = _structure_aware_clearance_regions(
                    family, topology_id, route, obstacles,
                )
            else:
                regions = []
            reference_poses = [
                _route_pose_at_fraction(route, fraction)
                for fraction in (0.40, 0.60)
            ]
            travel3, lateral3, _, region_quaternion = _route_sampling_frame(route)
            travel = travel3[:2]
            lateral = lateral3[:2]
            if not regions:
                for index, reference_pose in enumerate(reference_poses):
                    center_pose = list(reference_pose)
                    size_xyz = [0.58, 0.58, 0.48]
                    mode_constraint = "route_local_neighborhood"
                    attitude_jitter = [5.0, 4.0, 8.0]
                    if separator is not None and topology_id in {
                        "above", "below", "left", "right",
                    }:
                        separator_center, separator_rotation, separator_half = (
                            _obstacle_obb(separator)
                        )
                        separator_travel_half = float(np.sum(
                            np.abs(travel3 @ separator_rotation) * separator_half
                        ))
                        separator_lateral_half = float(np.sum(
                            np.abs(lateral3 @ separator_rotation) * separator_half
                        ))
                        vehicle_lateral_half = (
                            max(ROBOT.size[0], ROBOT.size[1]) / 2
                            + ROBOT.safety_margin
                        )
                        local_lateral = float(
                            (np.asarray(center_pose[:3]) - separator_center)
                            @ lateral3
                        )
                        longitudinal_size = min(
                            0.95, max(0.58, 0.65 * (2 * separator_travel_half))
                        )
                        if topology_id in {"above", "below"}:
                            lateral_size = min(
                                3.10,
                                max(0.90, 2 * separator_lateral_half + 0.55),
                            )
                            center_pose[0] -= local_lateral * lateral3[0]
                            center_pose[1] -= local_lateral * lateral3[1]
                            if topology_id == "above":
                                z_low = max(
                                    float(reference_pose[2]) - 0.12,
                                    float(separator_center[2] + separator_half[2])
                                    + ROBOT.size[2] / 2 + ROBOT.safety_margin + 0.05,
                                )
                                z_high = SAMPLING_BOUNDS["max"][2] - 0.10
                            else:
                                z_low = SAMPLING_BOUNDS["min"][2] + 0.10
                                z_high = min(
                                    float(reference_pose[2]) + 0.12,
                                    float(separator_center[2] - separator_half[2])
                                    - ROBOT.size[2] / 2 - ROBOT.safety_margin - 0.04,
                                )
                            if z_high <= z_low + 0.08:
                                z_low = float(reference_pose[2]) - 0.04
                                z_high = float(reference_pose[2]) + 0.04
                            center_pose[2] = 0.5 * (z_low + z_high)
                            size_xyz = [
                                longitudinal_size, lateral_size, z_high - z_low,
                            ]
                            mode_constraint = "vertical_bypass_slab"
                            attitude_jitter = (
                                [6.0, 5.0, 10.0]
                                if topology_id == "above" else [3.0, 3.0, 6.0]
                            )
                        else:
                            direction_sign = 1.0 if topology_id == "left" else -1.0
                            # Never sample closer to the separator than the
                            # collision-certified route (apart from a tiny
                            # tolerance).  The remaining interval expands
                            # outward toward the workspace boundary.
                            inner = max(
                                separator_lateral_half + vehicle_lateral_half + 0.10,
                                abs(local_lateral) - 0.04,
                            )

                            def ray_limit() -> float:
                                values = []
                                for axis in range(2):
                                    component = direction_sign * lateral[axis]
                                    if abs(float(component)) < 1e-9:
                                        continue
                                    boundary = (
                                        SAMPLING_BOUNDS["max"][axis]
                                        if component > 0 else SAMPLING_BOUNDS["min"][axis]
                                    )
                                    values.append(
                                        (boundary - separator_center[axis]) / component
                                    )
                                positive = [value for value in values if value > 0.0]
                                return min(positive) - 0.12 if positive else inner + 0.30

                            outer = max(inner + 0.22, ray_limit())
                            lateral_center = direction_sign * 0.5 * (inner + outer)
                            lateral_size = outer - inner
                            delta = lateral_center - local_lateral
                            center_pose[0] += delta * lateral3[0]
                            center_pose[1] += delta * lateral3[1]
                            z_low = max(
                                SAMPLING_BOUNDS["min"][2] + 0.14,
                                float(reference_pose[2]) - 0.72,
                            )
                            z_high = min(
                                SAMPLING_BOUNDS["max"][2] - 0.12,
                                float(reference_pose[2]) + 0.62,
                            )
                            center_pose[2] = 0.5 * (z_low + z_high)
                            size_xyz = [
                                longitudinal_size, lateral_size, z_high - z_low,
                            ]
                            mode_constraint = "lateral_bypass_slab"
                            attitude_jitter = [7.0, 6.0, 10.0]
                    regions.append({
                        "region_id": f"{topology_id}_portal_{index}",
                        "center_pose": center_pose,
                        "size_xyz": [round(float(value), 6) for value in size_xyz],
                        "quaternion_wxyz": region_quaternion,
                        "orientation_sampling": "bounded_reference_rpy_jitter",
                        "orientation_rpy_jitter_deg": attitude_jitter,
                        "mode_constraint": mode_constraint,
                    })
            policy = "region_biased_sampling"
            # Broad route-local regions are intentionally allowed to overlap
            # nearby clearance shells, but their nominal center must remain a
            # valid biased sample.  If a newly inserted shaper occupies the
            # analytically chosen center, project only the center back to the
            # closest pose on the surviving certificate; retain the broad
            # region extents so the sampler is not converted into a waypoint.
            for region in regions:
                if pose_is_free(region["center_pose"], obstacles):
                    continue
                original_center = np.asarray(
                    region["center_pose"][:3], dtype=np.float64,
                )
                replacement = min(
                    route,
                    key=lambda pose: float(np.linalg.norm(
                        np.asarray(pose[:3], dtype=np.float64) - original_center
                    )),
                )
                region["center_pose"] = list(replacement)
                region["center_projected_to_certificate"] = True
        guides.append({
            "id": topology_id,
            "policy": policy,
            "sampled_waypoint_regions": regions,
            "fixed_waypoints": fixed_waypoints,
            "terminal_perturbation_allowed": False,
            "coherent_local_axes": [1, 2],
            **({
                "uniform_state_sampling_probability": 0.30,
                "regional_state_sampling_probability": 0.70,
                "region_samples_are_waypoints": False,
            } if policy == "region_biased_sampling" else {}),
            **({"strict_mode_constraint": strict_constraint}
               if strict_constraint else {}),
        })
    return guides


def _fill_variation_props(
    obstacles: list[dict[str, Any]],
    topology_routes: list[tuple[str, list[list[float]]]],
    p: SceneParameters, rng: random.Random,
) -> tuple[list[tuple[str, list[list[float]]]], dict[str, Any]]:
    """Add grounded props that have a measured task-space function.

    The old filler protected every authored route pose.  In a multi-route
    scene that left only corners and the volume below the routes, so asking
    for more obstacles mostly produced attention tokens that could not affect
    the task.  This incremental packer instead uses the route bank as a
    *survival constraint*:

    - route selectors may invalidate an old certificate while the family
      still retains its minimum number of certified routes;
    - clearance shapers must preserve the route bank, but intersect its
      0.42 m inflated URDF clearance shell;
    - only a small explicit quota is allowed to be a distractor.

    The returned route bank contains only certificates that remain valid in
    the final scene.  It is deliberately a finite certificate bank rather
    than a claim that all homotopy classes have been enumerated; subsequent
    OMPL planning remains free to discover additional routes.
    """
    requested_props = p.obstacle_count - len(obstacles)
    initial_route_count = len(topology_routes)
    minimum_surviving_routes = (
        2
        if initial_route_count >= 2 and p.family in (
            GUARANTEED_MULTI_ROUTE_FAMILIES | OPTIONAL_MULTI_ROUTE_FAMILIES
        )
        else 1
    )
    if requested_props <= 0:
        return topology_routes, {
            "generated_prop_count": 0,
            "route_selector_count": 0,
            "clearance_shaper_count": 0,
            "distractor_count": 0,
            "effective_prop_count": 0,
            "effective_prop_ratio": 1.0,
            "initial_route_mode_count": initial_route_count,
            "surviving_route_mode_count": initial_route_count,
            "minimum_surviving_route_mode_count": minimum_surviving_routes,
            "blocked_route_modes": [],
        }
    current_routes = list(topology_routes)
    required_route_modes = (
        {"through"} if p.family == "frame_doorway"
        else {topology_routes[0][0]}
        if p.family == "orientation_sensitive_passage"
        else set()
    )
    prop_index = 0
    effective_size_max = max(p.size_min, min(p.size_max, 1.0))
    route_collision_cache: dict[str, dict[str, Any]] = {}
    for topology_id, candidate_route in topology_routes:
        dense_route = _interpolate_route(candidate_route)
        collision_routes = [dense_route]
        if p.family == "orientation_sensitive_passage":
            tilted = [
                pose for pose in candidate_route[1:-1]
                if max(abs(value) for value in quaternion_roll_pitch_degrees(
                    pose[3:7]
                )) > 15.0
            ]
            if len(tilted) >= 2:
                # The planner reconnects these four strict poses directly.
                # Protect those segments in addition to the richer authored
                # route so a clearance shaper cannot invalidate the fallback.
                collision_routes.append(_interpolate_route([
                    candidate_route[0], tilted[0], tilted[-1],
                    candidate_route[-1],
                ]))
        route_collision_cache[topology_id] = {
            "normal": [
                primitive_obb
                for collision_route in collision_routes
                for pose in collision_route
                for primitive_obb in _robot_primitive_obbs(pose)
            ],
            "influence": [
                primitive_obb
                for collision_route in collision_routes
                for pose in collision_route
                for primitive_obb in _robot_primitive_obbs(
                    pose, ROBOT.safety_margin + 0.42,
                )
            ],
            # Terminal attitude is randomized only after prop placement.  A
            # wider endpoint bubble prevents a nearby prop from allowing the
            # level certificate but rejecting every non-level attitude.
            "terminal": [
                primitive_obb
                for pose in (candidate_route[0], candidate_route[-1])
                for primitive_obb in _robot_primitive_obbs(
                    pose, ROBOT.safety_margin + 0.46,
                )
            ],
        }
    role_counts = {
        "route_selector": 0,
        "clearance_shaper": 0,
        "distractor": 0,
    }
    blocked_route_modes: list[str] = []
    # A 15% distractor quota teaches the conditioner that not every token is
    # decisive, without letting nuisance geometry dominate the token budget.
    distractor_target = min(
        requested_props - 1 if requested_props > 1 else 0,
        int(round(0.15 * requested_props)),
    )
    maximum_distractors = int(math.floor(0.35 * requested_props))
    selector_target = min(
        max(0, initial_route_count - minimum_surviving_routes),
        max(1, int(round(0.30 * requested_props))),
    )

    def route_pose_and_lateral(
        route: list[list[float]], fraction: float,
    ) -> tuple[list[float], np.ndarray]:
        pose = _route_pose_at_fraction(route, fraction)
        before = np.asarray(
            _route_pose_at_fraction(route, max(0.02, fraction - 0.035))[:3],
            dtype=np.float64,
        )
        after = np.asarray(
            _route_pose_at_fraction(route, min(0.98, fraction + 0.035))[:3],
            dtype=np.float64,
        )
        tangent = after[:2] - before[:2]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            lateral = np.asarray([0.0, 1.0], dtype=np.float64)
        else:
            lateral = np.asarray([-tangent[1], tangent[0]]) / norm
        return pose, lateral

    def make_candidate(
        x: float, y: float, sx: float, sy: float, sz: float,
        functional_role: str,
    ) -> dict[str, Any]:
        candidate = _box(
            f"variation_prop_{prop_index:02d}",
            (x, y, sz / 2),
            (sx, sy, sz), yaw=rng.uniform(-math.pi, math.pi),
            role="secondary_obstacle", family=p.family,
        )
        candidate["functional_role"] = functional_role
        candidate["functional_height_class"] = (
            "tall_pillar" if sz >= 2.45
            else "medium_blocker" if sz >= 1.45
            else "low_overfly"
        )
        candidate["route_relevant"] = functional_role != "distractor"
        return candidate

    def geometrically_placeable(candidate: dict[str, Any]) -> bool:
        if any(
            obstacles_overlap(candidate, other, padding=0.055)
            for other in obstacles
        ):
            return False
        # Check the finally transformed OBB against the fixed workspace too;
        # packing itself is performed in the template's local frame.
        transformed = {
            **candidate,
            "pose": {
                "position": list(candidate["pose"]["position"]),
                "quaternion_wxyz": list(
                    candidate["pose"]["quaternion_wxyz"]
                ),
            },
            "size_xyz": list(candidate["size_xyz"]),
        }
        _transform_obstacle(transformed, p)
        center, rotation, half = _obstacle_obb(transformed)
        world_half = np.abs(rotation) @ half
        if not all(
            abs(float(center[axis])) + float(world_half[axis]) <= 3.98
            for axis in (0, 1)
        ):
            return False
        candidate_obb = _obstacle_obb(candidate)
        return not any(
            _obb_overlap(*robot_obb, *candidate_obb)
            for topology_id, _ in current_routes
            for robot_obb in route_collision_cache[topology_id]["terminal"]
        )

    def route_effect(
        candidate: dict[str, Any],
    ) -> tuple[
        list[tuple[str, list[list[float]]]], list[str], list[str]
    ]:
        survivors = []
        blocked = []
        near = []
        candidate_obb = _obstacle_obb(candidate)
        for topology_id, candidate_route in current_routes:
            cache = route_collision_cache[topology_id]
            if any(
                _obb_overlap(*robot_obb, *candidate_obb)
                for robot_obb in cache["normal"]
            ):
                blocked.append(topology_id)
                continue
            survivors.append((topology_id, candidate_route))
            if any(
                _obb_overlap(*robot_obb, *candidate_obb)
                for robot_obb in cache["influence"]
            ):
                near.append(topology_id)
        return survivors, blocked, near

    def route_survival_is_valid(
        survivors: list[tuple[str, list[list[float]]]],
    ) -> bool:
        if len(survivors) < minimum_surviving_routes:
            return False
        survivor_ids = {topology_id for topology_id, _ in survivors}
        if p.family == "multi_homotopy":
            return bool(survivor_ids & {"above", "below"}) and bool(
                survivor_ids & {"left", "right"}
            )
        return required_route_modes.issubset(survivor_ids)

    def accept_candidate(
        candidate: dict[str, Any],
        survivors: list[tuple[str, list[list[float]]]],
        blocked: list[str], near: list[str],
        functional_role: str,
    ) -> None:
        nonlocal current_routes, prop_index
        candidate["functional_role"] = functional_role
        candidate["route_relevant"] = functional_role != "distractor"
        candidate["influence_evidence"] = {
            "blocked_certificate_modes": list(blocked),
            "clearance_shell_modes": list(near),
            "clearance_shell_m": 0.42 if near else None,
        }
        obstacles.append(candidate)
        current_routes = survivors
        role_counts[functional_role] += 1
        blocked_route_modes.extend(blocked)
        prop_index += 1

    def place_route_selector(limit: int) -> bool:
        if len(current_routes) <= minimum_surviving_routes:
            return False
        for _ in range(limit):
            topology_id, target_route = rng.choice(current_routes)
            fraction = rng.uniform(0.22, 0.78)
            pose, lateral = route_pose_and_lateral(target_route, fraction)
            sx = rng.uniform(0.26, min(0.50, effective_size_max))
            sy = rng.uniform(0.26, min(0.50, effective_size_max))
            # Grounded and high enough to intersect the selected route's
            # vehicle envelope; never synthesize a floating blocker.
            sz = min(3.72, max(1.65, float(pose[2]) + rng.uniform(0.28, 0.72)))
            jitter = rng.uniform(-0.16, 0.16)
            candidate = make_candidate(
                float(pose[0]) + jitter * float(lateral[0]),
                float(pose[1]) + jitter * float(lateral[1]),
                sx, sy, sz, "route_selector",
            )
            if not geometrically_placeable(candidate):
                continue
            survivors, blocked, near = route_effect(candidate)
            if topology_id not in blocked or not blocked:
                continue
            if not route_survival_is_valid(survivors):
                continue
            accept_candidate(
                candidate, survivors, blocked, near, "route_selector",
            )
            return True
        return False

    def place_clearance_shaper(limit: int) -> bool:
        for _ in range(limit):
            topology_id, target_route = rng.choice(current_routes)
            fraction = rng.uniform(0.20, 0.80)
            pose, lateral = route_pose_and_lateral(target_route, fraction)
            direction = rng.choice((-1.0, 1.0))
            sx = rng.uniform(0.24, min(0.46, effective_size_max))
            sy = rng.uniform(0.24, min(0.46, effective_size_max))
            sz = min(3.60, max(1.50, float(pose[2]) + rng.uniform(0.08, 0.55)))
            # The offset is deliberately sampled broadly and then certified
            # geometrically below.  This remains valid for yawed routes and
            # the asymmetric URDF envelope without relying on an AABB radius.
            offset = rng.uniform(0.46, 1.18)
            along = rng.uniform(-0.18, 0.18)
            tangent = np.asarray([lateral[1], -lateral[0]])
            candidate = make_candidate(
                float(pose[0]) + direction * offset * float(lateral[0])
                + along * float(tangent[0]),
                float(pose[1]) + direction * offset * float(lateral[1])
                + along * float(tangent[1]),
                sx, sy, sz, "clearance_shaper",
            )
            if not geometrically_placeable(candidate):
                continue
            survivors, blocked, near = route_effect(candidate)
            # A shaper deforms the clearance field but does not invalidate an
            # existing certificate.  Accidental blockers are handled only by
            # the selector stage, where route survival is explicit.
            if blocked or topology_id not in near:
                continue
            accept_candidate(
                candidate, survivors, blocked, near, "clearance_shaper",
            )
            return True
        return False

    def place_distractor(limit: int) -> bool:
        for _ in range(limit):
            sx = rng.uniform(min(p.size_min, 0.28), min(0.62, effective_size_max))
            sy = rng.uniform(min(p.size_min, 0.28), min(0.62, effective_size_max))
            # Most explicit distractors are low and overflyable.  Keeping the
            # category visible in metadata makes its small quota auditable.
            sz = rng.uniform(0.72, 1.32)
            maximum_radius = 2.76 - 0.30 * max(sx, sy)
            radius = math.sqrt(rng.uniform(0.35 ** 2, maximum_radius ** 2))
            angle = rng.uniform(-math.pi, math.pi)
            candidate = make_candidate(
                radius * math.cos(angle), radius * math.sin(angle),
                sx, sy, sz, "distractor",
            )
            if not geometrically_placeable(candidate):
                continue
            survivors, blocked, near = route_effect(candidate)
            if blocked or near:
                continue
            accept_candidate(
                candidate, survivors, blocked, near, "distractor",
            )
            return True
        return False

    while len(obstacles) < p.obstacle_count:
        remaining = p.obstacle_count - len(obstacles)
        need_distractors = max(
            0, distractor_target - role_counts["distractor"]
        )
        if (
            role_counts["route_selector"] < selector_target
            and place_route_selector(700)
        ):
            continue
        if remaining > need_distractors and place_clearance_shaper(900):
            continue
        if (
            need_distractors
            and role_counts["distractor"] < maximum_distractors
            and place_distractor(900)
        ):
            continue
        # If a dense scene exhausts the planned mix, prefer another effective
        # shell obstacle.  Only then consume the remaining distractor budget.
        if place_clearance_shaper(2200):
            continue
        if (
            role_counts["distractor"] < maximum_distractors
            and place_distractor(2200)
        ):
            continue
        raise ValueError(
            f"could place only {len(obstacles)}/{p.obstacle_count} "
            "non-overlapping task-relevant obstacles without exceeding the "
            "35% distractor budget; reduce count or choose another seed"
        )

    effective_count = (
        role_counts["route_selector"] + role_counts["clearance_shaper"]
    )
    return current_routes, {
        "generated_prop_count": requested_props,
        "route_selector_count": role_counts["route_selector"],
        "clearance_shaper_count": role_counts["clearance_shaper"],
        "distractor_count": role_counts["distractor"],
        "maximum_distractor_ratio": 0.35,
        "effective_prop_count": effective_count,
        "effective_prop_ratio": round(effective_count / requested_props, 6),
        "initial_route_mode_count": initial_route_count,
        "surviving_route_mode_count": len(current_routes),
        "minimum_surviving_route_mode_count": minimum_surviving_routes,
        "blocked_route_modes": blocked_route_modes,
        "effectiveness_definition": (
            "blocks_a_certificate_or_intersects_0.42m_inflated_urdf_clearance_shell"
        ),
    }


def generate_scene(raw: dict[str, Any] | SceneParameters) -> dict[str, Any]:
    if isinstance(raw, SceneParameters):
        p, requested_ranges = raw, {}
    else:
        p, requested_ranges = sample_scene_parameters(raw)
    rng = random.Random(p.seed)
    obstacles = FAMILY_BUILDERS[p.family](p, rng)[:p.obstacle_count]
    start_region, goal_region = _endpoint_regions(rng)
    topology_routes: list[tuple[str, list[list[float]]]] = []
    if p.family == "multi_homotopy":
        candidates = _multi_homotopy_routes(obstacles, start_region, goal_region, rng)
        if candidates is not None:
            topology_routes = candidates
        route = topology_routes[0][1] if topology_routes else None
    elif p.family == "frame_doorway":
        candidates = _frame_route_modes(
            obstacles, start_region, goal_region, rng,
        )
        if candidates is not None:
            topology_routes = candidates
        route = topology_routes[0][1] if topology_routes else None
    elif p.family in {"sparse_obb_clutter", "central_block", "pillar_wall"}:
        separator = next(
            (item for item in obstacles if item.get("route_separator")), None
        )
        candidates = _diverse_side_routes(
            obstacles, start_region, goal_region, rng, separator,
            include_above=True,
        )
        if candidates is not None:
            topology_routes = candidates
        route = topology_routes[0][1] if topology_routes else None
    elif p.family == "staggered_corridor":
        candidates = _diverse_side_routes(
            obstacles, start_region, goal_region, rng,
            allow_astar=False,
        )
        if candidates is not None:
            topology_routes = candidates
        route = topology_routes[0][1] if topology_routes else None
        if route is None and p.family in OPTIONAL_MULTI_ROUTE_FAMILIES:
            route = _find_upright_route(
                obstacles, start_region, goal_region, rng,
            )
    elif p.family == "mixed_industrial":
        candidates = _mixed_industrial_routes(
            obstacles, start_region, goal_region, rng,
        )
        if candidates is not None:
            topology_routes = candidates
            # Keep a genuinely free-space-discovered alternative when the
            # deterministic under/over constructions expose only one mode.
            # This route shares the exact task endpoints and is subsequently
            # protected from variation props like every authored candidate.
            astar_route = _a_star_route(
                topology_routes[0][1][0][:3],
                topology_routes[0][1][-1][:3],
                obstacles,
            )
            if astar_route is not None:
                reference = topology_routes[0][1]
                deviation = max(
                    float(np.linalg.norm(
                        np.asarray(_route_pose_at_fraction(astar_route, fraction)[:3])
                        - np.asarray(_route_pose_at_fraction(reference, fraction)[:3])
                    ))
                    for fraction in (0.25, 0.50, 0.75)
                )
                if deviation >= 0.35:
                    topology_routes.append(("free_space_alternative", astar_route))
        route = topology_routes[0][1] if topology_routes else None
    elif p.family == "orientation_sensitive_passage":
        route = _orientation_route(obstacles, start_region, goal_region, rng)
    elif p.family == "wall_protrusion_bracket":
        candidates = _bracket_dogleg_routes(
            obstacles, start_region, goal_region, rng,
        )
        if candidates is not None:
            topology_routes = candidates
        route = topology_routes[0][1] if topology_routes else None
    elif p.family == "narrow_passage":
        route = _portal_route(obstacles, start_region, goal_region, rng)
    else:
        route = _find_upright_route(obstacles, start_region, goal_region, rng)
    if route is None:
        raise ValueError(
            f"template core {p.family!r} has no conservative URDF-envelope route; choose another seed or wider valid range"
        )
    if (
        p.family in GUARANTEED_MULTI_ROUTE_FAMILIES
        and len(topology_routes) < 2
    ):
        raise ValueError(
            f"template core {p.family!r} could not preserve two URDF-feasible route modes"
        )
    if not topology_routes:
        topology_routes = [("primary", route)]
    route = topology_routes[0][1]
    topology_routes, obstacle_function_summary = _fill_variation_props(
        obstacles, topology_routes, p, rng,
    )
    route = topology_routes[0][1]
    # Place props against the geometric corridors first, then choose terminal
    # attitudes against the complete scene.  Inflating every corridor by a
    # random 70-degree endpoint pitch during packing needlessly removed most
    # useful tall-box locations.
    topology_routes = _randomize_terminal_attitudes(topology_routes, obstacles, rng)
    route = topology_routes[0][1]
    if not all(route_is_free(candidate, obstacles) for _, candidate in topology_routes):
        raise ValueError("internal error: secondary obstacles invalidated the feasibility route")
    for obstacle in obstacles:
        _transform_obstacle(obstacle, p)
    topology_routes = [
        (topology_id, [_transform_pose(pose, p) for pose in candidate])
        for topology_id, candidate in topology_routes
    ]
    route = topology_routes[0][1]
    expert_planning_guides = _expert_planning_guides(
        p.family, topology_routes, obstacles,
    )
    start_region = _transform_region(start_region, p)
    goal_region = _transform_region(goal_region, p)
    environment_id = f"{p.family}_seed_{p.seed}"
    scene = {
        "schema_version": "free_flight_scene_v001",
        "environment_id": environment_id,
        "map_id": environment_id,
        "generation_family": p.family,
        "generation_seed": p.seed,
        "generation_parameters": {
            key: getattr(p, key) for key in p.__dataclass_fields__
        },
        "requested_parameter_ranges": requested_ranges,
        "route_mode_policy": (
            "guaranteed_multi"
            if p.family in GUARANTEED_MULTI_ROUTE_FAMILIES
            else "optional_multi"
            if p.family in OPTIONAL_MULTI_ROUTE_FAMILIES
            else "single_passage"
        ),
        "verified_route_mode_count": len(topology_routes),
        "obstacle_function_summary": obstacle_function_summary,
        "units": {"length": "meter", "angle": "radian"},
        "coordinate_frame": {"name": "world", "handedness": "right", "up_axis": "Z"},
        "bounds": WORKSPACE_BOUNDS,
        "sampling_space": {
            "position_bounds": SAMPLING_BOUNDS,
            "orientation": {"space": "SO3"},
        },
        "robot_reference": {
            "urdf": str(URDF_PATH.relative_to(PROJECT_DIR)),
            "base_link": "base_link",
            "collision_object_names": list(ROBOT.collision_names),
            "collision_aabb_local": {
                "min": list(ROBOT.minimum), "max": list(ROBOT.maximum),
                "center": list(ROBOT.center), "size_xyz": list(ROBOT.size),
            },
            "safety_margin_m": ROBOT.safety_margin,
            "flight_attitude_limits_deg": {
                "roll": [-MAX_FLIGHT_ROLL_DEG, MAX_FLIGHT_ROLL_DEG],
                "pitch": [-MAX_FLIGHT_PITCH_DEG, MAX_FLIGHT_PITCH_DEG],
            },
            "collision_primitives": [{
                "name": primitive.name,
                "type": primitive.kind,
                "local_pose": {
                    "position": list(primitive.local_position),
                    "quaternion_wxyz": list(primitive.local_quaternion_wxyz),
                },
                "half_extents": list(primitive.half_extents),
            } for primitive in ROBOT.primitives],
        },
        "obstacles": [
            _box("floor", (0, 0, -0.08), (8, 8, 0.16), role="floor", family=p.family),
            *obstacles,
        ],
        "task_sampling": {
            "start_region": start_region,
            "goal_region": goal_region,
            "selection": "collision_free_rejection_sampling_with_connectivity_precheck",
            "regions_are_not_fixed_points": True,
        },
        "precheck_pairs": [{
            "pair_id": "primary",
            "start_pose": route[0],
            "goal_pose": route[-1],
            "sampled_from_regions": True,
        }],
        "expert_planning_guides": expert_planning_guides,
        "expert_route_templates": [{
            "id": topology_id,
            "route_poses": candidate,
            "hard_constraint": "practical_route_mode",
            "signature_proxy": {
                "separator_axis": "x",
                "class": topology_id,
                "route_axis": (
                    "vertical" if topology_id in {"above", "below"}
                    else "lateral"
                ),
            },
        } for topology_id, candidate in topology_routes] if len(topology_routes) > 1 else [],
        "feasibility_certificate": {
            "method": "conservative_urdf_aabb_route_v001",
            "status": "feasible",
            "safety_margin_m": ROBOT.safety_margin,
            "route_poses": route,
            "dense_check_spacing_m": 0.09,
            "maximum_angular_step_deg": 5.0,
            "requires_attitude_change": p.family == "orientation_sensitive_passage",
            "protected_route_mode_count": len(topology_routes),
            "note": "Conservative AABB certificate; formal datasets should still run full COAL/OMPL verification.",
        },
        "cross_attention_contract": {
            "schema_version": "box_geometry_v001",
            "maximum_obstacles": MAX_OBSTACLES,
            "feature_order": ["x", "y", "z", "size_x", "size_y", "size_z", "qw", "qx", "qy", "qz"],
            "floor_excluded": True,
        },
    }
    issues = validate_scene(scene)
    if issues:
        raise ValueError("generated scene failed validation: " + "; ".join(issues))
    return scene


def generate_batch(raw: dict[str, Any]) -> dict[str, Any]:
    """Generate a deterministic, randomized bank spanning selected families."""
    variants = int(raw.get("variants_per_family", 10))
    if not 1 <= variants <= 100:
        raise ValueError("variants_per_family must be in [1, 100]")
    base_seed = int(raw.get("base_seed", 1))
    families = raw.get("families", list(FAMILY_BUILDERS))
    if not isinstance(families, list) or not families:
        raise ValueError("families must be a nonempty list")
    if any(family not in FAMILY_BUILDERS for family in families):
        raise ValueError("families contains an unknown scene family")
    maximum_count = int(raw.get("obstacle_count_max", 10))
    if not 1 <= maximum_count <= MAX_OBSTACLES:
        raise ValueError(f"obstacle_count_max must be in [1, {MAX_OBSTACLES}]")
    size_min = float(raw.get("size_min", 0.45))
    size_max = float(raw.get("size_max", 1.00))
    count_minimum = int(raw.get("obstacle_count_min", 5))
    requested_gap_min = raw.get("gap_width_min")
    requested_gap_max = raw.get("gap_width_max")
    rng = random.Random(base_seed)
    scenes = []
    for family in families:
        if maximum_count < FAMILY_MINIMUMS[family]:
            raise ValueError(
                f"obstacle_count_max must be at least {FAMILY_MINIMUMS[family]} for {family}"
            )
        for variant in range(variants):
            chosen: dict[str, Any] | None = None
            for _ in range(40):
                parameters = {
                    "family": family,
                    "seed": rng.randrange(1, 2**31),
                    "sample_ranges": True,
                    "obstacle_count_min": max(FAMILY_MINIMUMS[family], count_minimum),
                    "obstacle_count_max": maximum_count,
                    "size_min": size_min,
                    "size_max": size_max,
                    "global_yaw_min": float(raw.get("global_yaw_min", -35)),
                    "global_yaw_max": float(raw.get("global_yaw_max", 35)),
                    "translation_max": float(raw.get("translation_max", 0.3)),
                }
                if requested_gap_min is not None:
                    parameters["gap_width_min"] = float(requested_gap_min)
                if requested_gap_max is not None:
                    parameters["gap_width_max"] = float(requested_gap_max)
                try:
                    candidate = generate_scene(parameters)
                    if not validate_scene(candidate):
                        chosen = candidate
                        break
                except ValueError:
                    continue
            if chosen is None:
                raise ValueError(f"could not generate a valid {family} variant after 40 attempts")
            chosen["environment_id"] = f"{family}_{variant:04d}_seed_{chosen['generation_seed']}"
            chosen["map_id"] = chosen["environment_id"]
            scenes.append(chosen)
    return {
        "schema_version": "free_flight_scene_bank_v001",
        "base_seed": base_seed,
        "variants_per_family": variants,
        "families": families,
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def obstacle_tokens(environment: dict[str, Any]) -> tuple[list[list[float]], list[bool]]:
    """Encode exactly the token/mask representation consumed by ConditionalDiT."""
    active = [
        item for item in environment.get("obstacles", [])
        if item.get("collision", False) and item.get("type") == "box"
        and item.get("role") != "floor"
    ]
    if len(active) > MAX_OBSTACLES:
        raise ValueError(f"environment has {len(active)} obstacles; maximum is {MAX_OBSTACLES}")
    tokens = np.zeros((MAX_OBSTACLES, 10), dtype=np.float32)
    mask = np.zeros(MAX_OBSTACLES, dtype=bool)
    for index, item in enumerate(active):
        tokens[index] = np.asarray([
            *item["pose"]["position"], *item["size_xyz"],
            *item["pose"]["quaternion_wxyz"],
        ], dtype=np.float32)
        mask[index] = True
    return tokens.tolist(), mask.tolist()


def validate_scene(environment: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    active = [item for item in environment.get("obstacles", []) if item.get("role") != "floor"]
    if len(active) > MAX_OBSTACLES:
        issues.append(f"障碍物数量 {len(active)} 超过 cross-attention 上限 {MAX_OBSTACLES}")
    identifiers: set[str] = set()
    for index, item in enumerate(active):
        label = str(item.get("id", f"obstacle[{index}]"))
        if label in identifiers:
            issues.append(f"重复障碍物 id: {label}")
        identifiers.add(label)
        if item.get("type") != "box":
            issues.append(f"{label}: cross-attention 当前仅支持 box")
            continue
        try:
            position = [float(value) for value in item["pose"]["position"]]
            size = [float(value) for value in item["size_xyz"]]
            quat = np.asarray(item["pose"]["quaternion_wxyz"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            issues.append(f"{label}: 几何字段不完整")
            continue
        if len(position) != 3 or len(size) != 3 or quat.shape != (4,):
            issues.append(f"{label}: position/size/quaternion 维度错误")
            continue
        if any(value <= 0 for value in size):
            issues.append(f"{label}: size 必须为正数")
        if not math.isclose(float(np.linalg.norm(quat)), 1.0, abs_tol=1e-3):
            issues.append(f"{label}: quaternion 未归一化")
        # Full-quaternion OBB projection, including the rolled passage frame.
        world_half = np.abs(quaternion_matrix(quat.tolist())) @ (np.asarray(size) / 2)
        if position[0] - world_half[0] < -4 or position[0] + world_half[0] > 4 or position[1] - world_half[1] < -4 or position[1] + world_half[1] > 4:
            issues.append(f"{label}: XY 包围盒超出固定场景边界")
        if position[2] - world_half[2] < -1e-5 or position[2] + world_half[2] > 4:
            issues.append(f"{label}: Z 包围盒超出固定场景边界")
        if (
            position[2] - world_half[2] > 0.03
            and not item.get("physical_support")
        ):
            issues.append(f"{label}: 悬空障碍物缺少地面立柱或挂载说明")
    for first_index, first in enumerate(active):
        for second in active[first_index + 1:]:
            shared_group = first.get("assembly_group")
            if shared_group and shared_group == second.get("assembly_group"):
                continue
            if obstacles_overlap(first, second):
                issues.append(f"{first['id']} 与 {second['id']} 发生体积重叠")
    certificate = environment.get("feasibility_certificate")
    if isinstance(certificate, dict) and certificate.get("route_poses"):
        route_poses = certificate["route_poses"]
        if not route_is_free(route_poses, active):
            issues.append("URDF 包络可行路线证书已被障碍物阻断")
        dense_route = _interpolate_route(route_poses)
        if not all(
            attitude_is_within_flight_limits(pose[3:7]) for pose in dense_route
        ):
            issues.append("可行路线证书超出 roll/pitch 飞行姿态限制")
        try:
            positions = np.asarray(route_poses, dtype=np.float64)[:, :3]
            position_bounds = environment["sampling_space"]["position_bounds"]
            lower = np.asarray(position_bounds["min"], dtype=np.float64)
            upper = np.asarray(position_bounds["max"], dtype=np.float64)
            if np.any(positions < lower) or np.any(positions > upper):
                issues.append("可行路线证书超出 OMPL position bounds")
        except (KeyError, TypeError, ValueError, IndexError):
            issues.append("可行路线证书或 OMPL position bounds 格式错误")
        if certificate.get("requires_attitude_change"):
            rotations = [pose[3:7] for pose in route_poses]
            if not any(abs(float(q[1])) > 0.18 or abs(float(q[2])) > 0.18 for q in rotations):
                issues.append("姿态敏感通道缺少实际 roll/pitch 变化")
    templates = environment.get("expert_route_templates", [])
    if templates:
        expected_endpoints = None
        for template in templates:
            route_poses = template.get("route_poses", [])
            if not route_poses or not route_is_free(route_poses, active):
                issues.append(
                    f"路线模式 {template.get('id', '?')} 的 URDF 证书已失效"
                )
                continue
            endpoints = np.asarray(
                [route_poses[0], route_poses[-1]], dtype=np.float64,
            )
            if expected_endpoints is None:
                expected_endpoints = endpoints
            elif not np.allclose(endpoints, expected_endpoints, atol=1e-7):
                issues.append("多路线模式没有共享完全相同的 start/goal")
    function_summary = environment.get("obstacle_function_summary")
    if isinstance(function_summary, dict):
        generated_props = [
            item for item in active
            if item.get("role") == "secondary_obstacle"
        ]
        role_counts = {
            role: sum(
                item.get("functional_role") == role
                for item in generated_props
            )
            for role in ("route_selector", "clearance_shaper", "distractor")
        }
        expected_fields = {
            "route_selector": "route_selector_count",
            "clearance_shaper": "clearance_shaper_count",
            "distractor": "distractor_count",
        }
        if int(function_summary.get("generated_prop_count", -1)) != len(generated_props):
            issues.append("障碍功能统计与随机填充物数量不一致")
        for role, field in expected_fields.items():
            if int(function_summary.get(field, -1)) != role_counts[role]:
                issues.append(f"障碍功能统计 {field} 与障碍元数据不一致")
        effective_count = (
            role_counts["route_selector"] + role_counts["clearance_shaper"]
        )
        if int(function_summary.get("effective_prop_count", -1)) != effective_count:
            issues.append("有效障碍物统计与障碍元数据不一致")
        if generated_props:
            measured_ratio = effective_count / len(generated_props)
            reported_ratio = float(
                function_summary.get("effective_prop_ratio", -1.0)
            )
            if not math.isclose(measured_ratio, reported_ratio, abs_tol=1e-6):
                issues.append("有效障碍物比例统计不一致")
            maximum_distractor_ratio = float(
                function_summary.get("maximum_distractor_ratio", 0.35)
            )
            if role_counts["distractor"] > math.floor(
                maximum_distractor_ratio * len(generated_props)
            ):
                issues.append("干扰物比例超过场景生成预算")
        minimum_modes = int(
            function_summary.get("minimum_surviving_route_mode_count", 1)
        )
        surviving_modes = int(
            function_summary.get("surviving_route_mode_count", 0)
        )
        if surviving_modes < minimum_modes:
            issues.append("功能障碍淘汰了过多可行路线证书")
        for item in generated_props:
            role = item.get("functional_role")
            evidence = item.get("influence_evidence", {})
            blocked = evidence.get("blocked_certificate_modes", [])
            near = evidence.get("clearance_shell_modes", [])
            if role == "route_selector" and not blocked:
                issues.append(f"{item['id']}: 路径选择障碍缺少证书阻断证据")
            elif role == "clearance_shaper" and not near:
                issues.append(f"{item['id']}: 净空塑形障碍缺少影响壳层证据")
            elif role == "distractor" and (blocked or near):
                issues.append(f"{item['id']}: 干扰物实际影响了证书路线")
    guides = environment.get("expert_planning_guides", [])
    if not guides:
        issues.append("场景缺少专家规划引导区域/固定点")
    for guide in guides:
        fixed_waypoints = guide.get("fixed_waypoints", [])
        if not fixed_waypoints or not route_is_free(fixed_waypoints, active):
            issues.append(f"专家引导 {guide.get('id', '?')} 的固定回退路线无效")
        policy = guide.get("policy")
        regions = guide.get("sampled_waypoint_regions", [])
        if policy == "region_biased_sampling" and not regions:
            issues.append(f"专家引导 {guide.get('id', '?')} 缺少采样区域")
        if policy == "region_biased_sampling":
            if guide.get("uniform_state_sampling_probability") != 0.30:
                issues.append(f"专家引导 {guide.get('id', '?')} 全局状态采样比例不是 30%")
            if guide.get("regional_state_sampling_probability") != 0.70:
                issues.append(f"专家引导 {guide.get('id', '?')} 区域状态采样比例不是 70%")
            if guide.get("region_samples_are_waypoints") is not False:
                issues.append(f"专家引导 {guide.get('id', '?')} 错误地把区域样本设为 waypoint")
        if policy == "fixed_waypoints_required" and regions:
            issues.append(f"严格专家引导 {guide.get('id', '?')} 不应包含随机区域")
        for region in regions:
            try:
                center_pose = region["center_pose"]
                size = np.asarray(region["size_xyz"], dtype=np.float64)
                region_quaternion = np.asarray(
                    region.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
                    dtype=np.float64,
                )
                if (
                    len(center_pose) != 7 or size.shape != (3,)
                    or np.any(size <= 0.0)
                    or region_quaternion.shape != (4,)
                    or not math.isclose(
                        float(np.linalg.norm(region_quaternion)),
                        1.0, abs_tol=1e-3,
                    )
                ):
                    raise ValueError
                if not pose_is_free(center_pose, active):
                    issues.append(
                        f"专家引导区域 {region.get('region_id', '?')} 中心发生碰撞"
                    )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    f"专家引导区域 {region.get('region_id', '?')} 格式错误"
                )
    return issues


def main() -> None:
    """Backward-compatible entry point for the standalone collector UI."""

    # When this file is executed directly, expose the already-loaded domain
    # module under its import name.  The free-flight task plugin can then reuse
    # it without evaluating the 3,000-line geometry module a second time.
    import sys

    sys.modules.setdefault("obstacle_scene_builder", sys.modules[__name__])
    from expert_trajectory_collector.cli import main as collector_main

    collector_main(default_task="free_flight")


if __name__ == "__main__":
    main()
