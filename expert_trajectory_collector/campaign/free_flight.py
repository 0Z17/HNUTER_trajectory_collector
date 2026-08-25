"""Free-flight condition sampling and one-environment worker implementation."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import random
import time
import traceback
from typing import Any

from obstacle_scene_builder import (
    FAMILY_MINIMUMS,
    _interpolate_route,
    _sample_region,
    attitude_is_within_flight_limits,
    generate_scene,
    route_is_free,
    rpy_quaternion,
    validate_scene,
)
from obstacle_scene_experts import (
    ConservativeURDFCollisionChecker,
    generate_expert_trajectories,
)

from .config import BatchConfig
from .io import atomic_write_json, read_json, save_expert_set


def derive_seed(base_seed: int, *parts: object) -> int:
    payload = "/".join((str(base_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _terminal_quaternion(rng: random.Random, base_yaw_deg: float, compact: bool) -> list[float]:
    roll_upper, pitch_upper = (18.0, 30.0) if compact else (40.0, 70.0)
    return rpy_quaternion(
        math.radians(rng.uniform(4.0, roll_upper) * rng.choice((-1, 1))),
        math.radians(rng.uniform(3.0, pitch_upper) * rng.choice((-1, 1))),
        math.radians(base_yaw_deg + rng.uniform(-18.0, 18.0)),
    )


def _replace_route_endpoints(
    route: list[list[float]], start_pose: list[float], goal_pose: list[float],
) -> None:
    route[0] = start_pose.copy()
    route[-1] = goal_pose.copy()


def sample_condition(
    base_scene: dict[str, Any], condition_index: int, seed: int,
) -> dict[str, Any]:
    """Sample new terminals while retaining all certified route interiors."""
    obstacles = [
        item for item in base_scene["obstacles"] if item.get("role") != "floor"
    ]
    start_region = base_scene["task_sampling"]["start_region"]
    goal_region = base_scene["task_sampling"]["goal_region"]
    base_yaw = float(base_scene.get("generation_parameters", {}).get("global_yaw_deg", 0.0))
    compact = base_scene.get("generation_family") in {
        "mixed_industrial", "wall_protrusion_bracket",
    }
    for attempt in range(160):
        rng = random.Random(f"condition/{seed}/{condition_index}/{attempt}")
        start_pose = [
            *_sample_region(start_region, rng),
            *_terminal_quaternion(rng, base_yaw, compact),
        ]
        goal_pose = [
            *_sample_region(goal_region, rng),
            *_terminal_quaternion(rng, base_yaw, compact),
        ]
        candidate = copy.deepcopy(base_scene)
        for guide in candidate.get("expert_planning_guides", []):
            _replace_route_endpoints(guide["fixed_waypoints"], start_pose, goal_pose)
        for template in candidate.get("expert_route_templates", []):
            _replace_route_endpoints(template["route_poses"], start_pose, goal_pose)
        certificate = candidate["feasibility_certificate"]["route_poses"]
        _replace_route_endpoints(certificate, start_pose, goal_pose)
        pair_id = f"condition_{condition_index:03d}"
        candidate["precheck_pairs"] = [{
            "pair_id": pair_id,
            "start_pose": start_pose,
            "goal_pose": goal_pose,
            "sampled_from_regions": True,
            "sampling_seed": seed,
            "sampling_attempt": attempt,
        }]
        candidate["condition_id"] = (
            f"{base_scene['environment_id']}__condition_{condition_index:03d}"
        )
        candidate["task_sampling"]["condition_seed"] = seed
        routes = [certificate] + [
            guide["fixed_waypoints"]
            for guide in candidate.get("expert_planning_guides", [])
        ] + [
            template["route_poses"]
            for template in candidate.get("expert_route_templates", [])
        ]
        if not all(
            route_is_free(route, obstacles)
            and all(attitude_is_within_flight_limits(pose[3:7]) for pose in _interpolate_route(route))
            for route in routes
        ):
            continue
        if not validate_scene(candidate):
            return candidate
    raise ValueError("could not sample collision-free start/goal condition after 160 attempts")


def _wait_while_paused(root: Path, progress_path: Path, progress: dict[str, Any]) -> None:
    announced = False
    while (root / "PAUSED").exists():
        if not announced:
            progress["worker_state"] = "paused"
            progress["updated_at_unix_s"] = time.time()
            atomic_write_json(progress_path, progress)
            announced = True
        time.sleep(0.25)
    if announced:
        progress["worker_state"] = "running"
        progress["updated_at_unix_s"] = time.time()
        atomic_write_json(progress_path, progress)


def _generate_environment(config: BatchConfig, environment_index: int) -> dict[str, Any]:
    family = config.families[environment_index % len(config.families)]
    minimum = max(FAMILY_MINIMUMS[family], config.obstacle_count_min)
    if minimum > config.obstacle_count_max:
        raise ValueError(
            f"{family} requires {minimum} obstacles but maximum is {config.obstacle_count_max}"
        )
    failures: list[str] = []
    for attempt in range(60):
        seed = derive_seed(config.base_seed, "environment", environment_index, attempt)
        try:
            scene = generate_scene({
                "family": family,
                "seed": seed,
                "sample_ranges": True,
                "obstacle_count_min": minimum,
                "obstacle_count_max": config.obstacle_count_max,
                "size_min": config.size_min_m,
                "size_max": config.size_max_m,
                "global_yaw_min": config.global_yaw_min_deg,
                "global_yaw_max": config.global_yaw_max_deg,
                "translation_max": config.translation_max_m,
            })
            scene["environment_id"] = f"env_{environment_index:06d}_{family}_seed_{seed}"
            scene["map_id"] = scene["environment_id"]
            scene["group_id"] = f"group_env_{environment_index:06d}"
            return scene
        except ValueError as error:
            failures.append(str(error))
    raise RuntimeError(
        f"failed to generate environment {environment_index} after 60 attempts; "
        f"last errors: {failures[-3:]}"
    )


def collect_environment(
    config_value: dict[str, Any], output_root_value: str, environment_index: int,
) -> dict[str, Any]:
    config = BatchConfig.from_dict(config_value)
    root = Path(output_root_value)
    environment_dir = root / "environments" / f"env_{environment_index:06d}"
    progress_path = environment_dir / "progress.json"
    environment_path = environment_dir / "environment.json"
    environment_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    progress = read_json(progress_path, {}) or {}
    progress.update({
        "schema_version": "expert_environment_progress_v001",
        "environment_index": environment_index,
        "worker_state": "running",
        "started_at_unix_s": progress.get("started_at_unix_s", started),
        "updated_at_unix_s": started,
        "condition_count": int(progress.get("condition_count", 0)),
        "accepted_path_count": int(progress.get("accepted_path_count", 0)),
        "planner_attempt_count": int(progress.get("planner_attempt_count", 0)),
        "failed_condition_count": int(progress.get("failed_condition_count", 0)),
    })
    atomic_write_json(progress_path, progress)
    try:
        environment = read_json(environment_path)
        if environment is None:
            environment = _generate_environment(config, environment_index)
            atomic_write_json(environment_path, environment)
        progress["environment_id"] = environment["environment_id"]
        progress["family"] = environment["generation_family"]
        checker = ConservativeURDFCollisionChecker(environment)
        progress["collision_backend"] = checker.backend_name
        condition_index = int(progress["condition_count"])
        accepted_total = int(progress["accepted_path_count"])
        while (
            accepted_total < config.paths_per_environment
            and condition_index < config.maximum_conditions_per_environment
        ):
            _wait_while_paused(root, progress_path, progress)
            condition_seed = derive_seed(
                config.base_seed, "condition", environment_index, condition_index,
            )
            condition_dir = environment_dir / "conditions" / f"condition_{condition_index:03d}"
            existing_metadata = read_json(condition_dir / "expert_metadata.json")
            if existing_metadata is not None:
                accepted = int(existing_metadata.get("accepted_count", 0))
                attempts = int(
                    existing_metadata.get("acceptance_pipeline", {})
                    .get("counts", {}).get("attempted", 0)
                )
            else:
                try:
                    condition = sample_condition(environment, condition_index, condition_seed)
                    requested = min(
                        config.experts_per_condition,
                        config.paths_per_environment - accepted_total,
                    )
                    expert_set = generate_expert_trajectories(
                        condition,
                        count=requested,
                        seed=derive_seed(condition_seed, "planner"),
                        solve_time=config.solve_time_s,
                        diversity_threshold_m=config.diversity_threshold_m,
                        planning_mode=config.planning_mode,
                        collision_checker=checker,
                        maximum_attempts=config.maximum_planner_attempts,
                    )
                    save_expert_set(
                        condition_dir, condition, expert_set,
                        config.training_path_points, checker,
                    )
                    accepted = int(expert_set.get("accepted_count", len(expert_set.get("experts", []))))
                    attempts = int(
                        expert_set.get("acceptance_pipeline", {})
                        .get("counts", {}).get("attempted", 0)
                    )
                except Exception as condition_error:
                    condition_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(condition_dir / "failure.json", {
                        "condition_index": condition_index,
                        "condition_seed": condition_seed,
                        "error": str(condition_error),
                        "traceback": traceback.format_exc(),
                    })
                    accepted, attempts = 0, 0
            accepted_total += accepted
            condition_index += 1
            progress.update({
                "worker_state": "running",
                "condition_count": condition_index,
                "accepted_path_count": accepted_total,
                "planner_attempt_count": int(progress["planner_attempt_count"]) + attempts,
                "failed_condition_count": int(progress["failed_condition_count"]) + (accepted == 0),
                "updated_at_unix_s": time.time(),
            })
            atomic_write_json(progress_path, progress)
        progress["worker_state"] = (
            "complete" if accepted_total >= config.paths_per_environment else "capacity_exhausted"
        )
        progress["completed_at_unix_s"] = time.time()
        progress["updated_at_unix_s"] = progress["completed_at_unix_s"]
        atomic_write_json(progress_path, progress)
        return progress
    except Exception as error:
        progress.update({
            "worker_state": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "updated_at_unix_s": time.time(),
        })
        atomic_write_json(progress_path, progress)
        return progress
