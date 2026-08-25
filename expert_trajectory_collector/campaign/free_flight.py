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
    MAX_FLIGHT_PITCH_DEG,
    MAX_FLIGHT_ROLL_DEG,
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


def _terminal_quaternion(
    rng: random.Random, base_yaw_deg: float, compact: bool,
    attitude_margin_deg: float,
) -> list[float]:
    roll_upper, pitch_upper = (
        (18.0, 30.0)
        if compact else (
            MAX_FLIGHT_ROLL_DEG - attitude_margin_deg,
            MAX_FLIGHT_PITCH_DEG - attitude_margin_deg,
        )
    )
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


def _route_with_endpoints(
    route: list[list[float]], start_pose: list[float], goal_pose: list[float],
) -> list[list[float]]:
    updated = copy.deepcopy(route)
    _replace_route_endpoints(updated, start_pose, goal_pose)
    return updated


def _route_is_valid(
    route: list[list[float]], obstacles: list[dict[str, Any]],
) -> bool:
    return (
        route_is_free(route, obstacles)
        and all(
            attitude_is_within_flight_limits(pose[3:7])
            for pose in _interpolate_route(route)
        )
    )


def _condition_mode_requirement(
    base_scene: dict[str, Any], retained_mode_ids: set[str],
) -> tuple[bool, int, str]:
    family = str(base_scene.get("generation_family", ""))
    policy = str(base_scene.get("route_mode_policy", "single_passage"))
    minimum = 2 if policy == "guaranteed_multi" else 1
    if len(retained_mode_ids) < minimum:
        return False, minimum, f"requires_at_least_{minimum}_route_modes"
    if family == "multi_homotopy" and not (
        retained_mode_ids & {"above", "below"}
        and retained_mode_ids & {"left", "right"}
    ):
        return False, minimum, "requires_vertical_and_lateral_route_modes"
    if family == "frame_doorway" and "through" not in retained_mode_ids:
        return False, minimum, "requires_through_route_mode"
    return True, minimum, "satisfied"


def _condition_with_valid_routes(
    base_scene: dict[str, Any], start_pose: list[float], goal_pose: list[float],
    *, condition_index: int, seed: int, sampling_attempt: int,
    sampling_elapsed_s: float,
) -> dict[str, Any] | None:
    """Build a condition from the subset of route modes valid for its terminals."""
    obstacles = [
        item for item in base_scene["obstacles"] if item.get("role") != "floor"
    ]
    original_guides = base_scene.get("expert_planning_guides", [])
    valid_guides: list[dict[str, Any]] = []
    dropped_mode_ids: list[str] = []
    for guide in original_guides:
        mode_id = str(guide.get("id", "primary"))
        route = _route_with_endpoints(
            guide.get("fixed_waypoints", []), start_pose, goal_pose,
        )
        if not _route_is_valid(route, obstacles):
            dropped_mode_ids.append(mode_id)
            continue
        updated_guide = copy.deepcopy(guide)
        updated_guide["fixed_waypoints"] = route
        valid_guides.append(updated_guide)

    retained_mode_ids = {str(guide.get("id", "primary")) for guide in valid_guides}
    requirement_met, minimum_modes, requirement = _condition_mode_requirement(
        base_scene, retained_mode_ids,
    )
    if not requirement_met:
        return None

    valid_templates: list[dict[str, Any]] = []
    for template in base_scene.get("expert_route_templates", []):
        mode_id = str(template.get("id", "primary"))
        if mode_id not in retained_mode_ids:
            continue
        route = _route_with_endpoints(
            template.get("route_poses", []), start_pose, goal_pose,
        )
        if not _route_is_valid(route, obstacles):
            continue
        updated_template = copy.deepcopy(template)
        updated_template["route_poses"] = route
        valid_templates.append(updated_template)

    candidate = copy.deepcopy(base_scene)
    candidate["expert_planning_guides"] = valid_guides
    candidate["expert_route_templates"] = valid_templates
    primary_guide = valid_guides[0]
    primary_mode_id = str(primary_guide.get("id", "primary"))
    candidate["feasibility_certificate"]["route_poses"] = copy.deepcopy(
        primary_guide["fixed_waypoints"]
    )
    candidate["feasibility_certificate"]["protected_route_mode_count"] = len(
        valid_guides
    )
    candidate["feasibility_certificate"]["condition_primary_route_mode"] = (
        primary_mode_id
    )
    candidate["verified_route_mode_count"] = len(valid_guides)
    pair_id = f"condition_{condition_index:03d}"
    candidate["precheck_pairs"] = [{
        "pair_id": pair_id,
        "start_pose": start_pose,
        "goal_pose": goal_pose,
        "sampled_from_regions": True,
        "sampling_seed": seed,
        "sampling_attempt": sampling_attempt,
    }]
    candidate["condition_id"] = (
        f"{base_scene['environment_id']}__condition_{condition_index:03d}"
    )
    candidate["task_sampling"]["condition_seed"] = seed
    candidate["condition_route_validation"] = {
        "policy": "retain_condition_valid_route_subset_v002",
        "all_environment_routes_required": False,
        "minimum_required_mode_count": minimum_modes,
        "requirement": requirement,
        "original_mode_ids": [
            str(guide.get("id", "primary")) for guide in original_guides
        ],
        "retained_mode_ids": [
            str(guide.get("id", "primary")) for guide in valid_guides
        ],
        "retained_template_mode_ids": [
            str(template.get("id", "primary")) for template in valid_templates
        ],
        "dropped_mode_ids": dropped_mode_ids,
        "sampling_attempt": sampling_attempt,
        "sampling_elapsed_s": sampling_elapsed_s,
    }
    return candidate


def sample_condition(
    base_scene: dict[str, Any], condition_index: int, seed: int, *,
    maximum_attempts: int = 160, timeout_s: float | None = None,
    terminal_attitude_margin_deg: float = 5.0,
) -> dict[str, Any]:
    """Sample terminals and retain only route modes valid for that condition."""
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    if timeout_s is not None and timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive when provided")
    start_region = base_scene["task_sampling"]["start_region"]
    goal_region = base_scene["task_sampling"]["goal_region"]
    base_yaw = float(base_scene.get("generation_parameters", {}).get("global_yaw_deg", 0.0))
    compact = base_scene.get("generation_family") in {
        "mixed_industrial", "wall_protrusion_bracket",
    }
    started = time.perf_counter()
    attempted = 0
    for attempt in range(maximum_attempts):
        if attempt and timeout_s is not None and time.perf_counter() - started >= timeout_s:
            break
        attempted += 1
        rng = random.Random(f"condition/{seed}/{condition_index}/{attempt}")
        start_pose = [
            *_sample_region(start_region, rng),
            *_terminal_quaternion(
                rng, base_yaw, compact, terminal_attitude_margin_deg,
            ),
        ]
        goal_pose = [
            *_sample_region(goal_region, rng),
            *_terminal_quaternion(
                rng, base_yaw, compact, terminal_attitude_margin_deg,
            ),
        ]
        elapsed = time.perf_counter() - started
        candidate = _condition_with_valid_routes(
            base_scene, start_pose, goal_pose,
            condition_index=condition_index, seed=seed,
            sampling_attempt=attempt, sampling_elapsed_s=elapsed,
        )
        if candidate is None:
            continue
        if not validate_scene(candidate):
            return candidate
    elapsed = time.perf_counter() - started
    budget = (
        f"{maximum_attempts} attempts"
        if timeout_s is None else f"{maximum_attempts} attempts/{timeout_s:.2f}s"
    )
    raise ValueError(
        "could not sample a start/goal condition with the required valid route "
        f"modes after {attempted} attempts in {elapsed:.3f}s (budget {budget})"
    )


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


def _precheck_environment_condition_sampling(
    config: BatchConfig, environment: dict[str, Any], environment_index: int,
) -> None:
    probe_count = config.environment_precheck_condition_count
    if probe_count == 0:
        environment["condition_sampling_precheck"] = {
            "enabled": False,
            "schema_version": "condition_sampling_precheck_v001",
        }
        return
    started = time.perf_counter()
    successes = 0
    attempted = 0
    failures: list[str] = []
    for probe_index in range(probe_count):
        attempted += 1
        try:
            sample_condition(
                environment, 1_000_000 + probe_index,
                derive_seed(
                    config.base_seed, "environment_precheck",
                    environment_index, probe_index,
                ),
                maximum_attempts=config.environment_precheck_max_attempts,
                timeout_s=config.environment_precheck_timeout_s,
                terminal_attitude_margin_deg=config.terminal_attitude_margin_deg,
            )
        except ValueError as error:
            failures.append(str(error))
        else:
            successes += 1
        remaining = probe_count - attempted
        if successes + remaining < config.environment_precheck_minimum_successes:
            break
    metadata = {
        "enabled": True,
        "schema_version": "condition_sampling_precheck_v001",
        "probe_count": probe_count,
        "attempted_probe_count": attempted,
        "successful_probe_count": successes,
        "minimum_successful_probe_count": (
            config.environment_precheck_minimum_successes
        ),
        "passed": successes >= config.environment_precheck_minimum_successes,
        "maximum_attempts_per_probe": config.environment_precheck_max_attempts,
        "timeout_s_per_probe": config.environment_precheck_timeout_s,
        "elapsed_s": time.perf_counter() - started,
        "recent_failures": failures[-3:],
    }
    environment["condition_sampling_precheck"] = metadata
    if not metadata["passed"]:
        raise ValueError(
            "environment condition-sampling precheck accepted only "
            f"{successes}/{attempted} probes; requires "
            f"{config.environment_precheck_minimum_successes}/{probe_count}"
        )


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
                "attitude_guard_deg": config.terminal_attitude_margin_deg,
            })
            scene["environment_id"] = f"env_{environment_index:06d}_{family}_seed_{seed}"
            scene["map_id"] = scene["environment_id"]
            scene["group_id"] = f"group_env_{environment_index:06d}"
            _precheck_environment_condition_sampling(
                config, scene, environment_index,
            )
            scene["campaign_environment_generation"] = {
                "schema_version": "campaign_environment_generation_v002",
                "accepted_attempt": attempt,
                "accepted_seed": seed,
                "rejected_attempt_count": len(failures),
                "recent_rejection_reasons": failures[-5:],
            }
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
        "schema_version": (
            "expert_environment_progress_v002"
            if config.schema_version == "expert_collection_campaign_config_v002"
            else "expert_environment_progress_v001"
        ),
        "environment_index": environment_index,
        "worker_state": "running",
        "started_at_unix_s": progress.get("started_at_unix_s", started),
        "updated_at_unix_s": started,
        "condition_count": int(progress.get("condition_count", 0)),
        "accepted_path_count": int(progress.get("accepted_path_count", 0)),
        "planner_attempt_count": int(progress.get("planner_attempt_count", 0)),
        "failed_condition_count": int(progress.get("failed_condition_count", 0)),
        "consecutive_condition_failure_count": int(
            progress.get("consecutive_condition_failure_count", 0)
        ),
        "condition_failure_reason_counts": dict(
            progress.get("condition_failure_reason_counts", {})
        ),
    })
    atomic_write_json(progress_path, progress)
    try:
        environment = read_json(environment_path)
        if environment is None:
            environment = _generate_environment(config, environment_index)
            atomic_write_json(environment_path, environment)
        progress["environment_id"] = environment["environment_id"]
        progress["family"] = environment["generation_family"]
        effective_experts_per_condition = (
            config.experts_per_condition_for_family(
                str(environment["generation_family"]),
            )
        )
        progress["experts_per_condition"] = effective_experts_per_condition
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
            failure_reason = None
            if existing_metadata is not None:
                accepted = int(existing_metadata.get("accepted_count", 0))
                attempts = int(
                    existing_metadata.get("acceptance_pipeline", {})
                    .get("counts", {}).get("attempted", 0)
                )
                if accepted == 0:
                    failure_reason = "no_experts_accepted"
            else:
                try:
                    condition = sample_condition(
                        environment, condition_index, condition_seed,
                        maximum_attempts=config.condition_sampling_max_attempts,
                        timeout_s=config.condition_sampling_timeout_s or None,
                        terminal_attitude_margin_deg=(
                            config.terminal_attitude_margin_deg
                        ),
                    )
                    requested = min(
                        effective_experts_per_condition,
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
                    failure_reason = (
                        "condition_sampling"
                        if str(condition_error).startswith(
                            "could not sample a start/goal condition"
                        )
                        else "expert_generation"
                    )
                    condition_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(condition_dir / "failure.json", {
                        "condition_index": condition_index,
                        "condition_seed": condition_seed,
                        "failure_reason": failure_reason,
                        "error": str(condition_error),
                        "traceback": traceback.format_exc(),
                    })
                    accepted, attempts = 0, 0
            accepted_total += accepted
            condition_index += 1
            consecutive_failures = (
                int(progress["consecutive_condition_failure_count"]) + 1
                if accepted == 0 else 0
            )
            failure_reason_counts = dict(
                progress["condition_failure_reason_counts"]
            )
            if accepted == 0:
                reason = failure_reason or "no_experts_accepted"
                failure_reason_counts[reason] = (
                    int(failure_reason_counts.get(reason, 0)) + 1
                )
            progress.update({
                "worker_state": "running",
                "condition_count": condition_index,
                "accepted_path_count": accepted_total,
                "planner_attempt_count": int(progress["planner_attempt_count"]) + attempts,
                "failed_condition_count": int(progress["failed_condition_count"]) + (accepted == 0),
                "consecutive_condition_failure_count": consecutive_failures,
                "condition_failure_reason_counts": failure_reason_counts,
                "updated_at_unix_s": time.time(),
            })
            atomic_write_json(progress_path, progress)
            if (
                accepted_total < config.paths_per_environment
                and consecutive_failures
                >= config.maximum_consecutive_condition_failures
            ):
                break
        progress["worker_state"] = (
            "complete" if accepted_total >= config.paths_per_environment else "capacity_exhausted"
        )
        progress["termination_reason"] = (
            "path_target_reached"
            if accepted_total >= config.paths_per_environment
            else "consecutive_condition_failure_limit"
            if int(progress["consecutive_condition_failure_count"])
            >= config.maximum_consecutive_condition_failures
            else "maximum_condition_count"
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
