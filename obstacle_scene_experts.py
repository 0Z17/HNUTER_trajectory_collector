"""Interactive expert planning for scenes produced by obstacle_scene_builder.

Ordinary experts use one global start-goal OMPL RRTConnect query whose C++
state sampler draws 70% of states from scene-authored guide regions and 30%
from the whole workspace. Region samples are never waypoint constraints.
Fixed waypoints are used only for strict orientation-coupled passages;
ordinary failures increase regional bias instead. Every result is collision
validated.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np

from obstacle_scene_builder import (
    MAX_FLIGHT_PITCH_DEG,
    MAX_FLIGHT_ROLL_DEG,
    ROBOT,
    URDF_PATH,
    attitude_is_within_flight_limits,
    pose_is_free,
    quaternion_roll_pitch_degrees,
)


class ConservativeURDFCollisionChecker:
    """OMPL adapter using COAL's full URDF primitives when available.

    The existing OBB implementation remains as a dependency-free fallback for
    scene generation and tests.  COAL additionally provides true signed
    distance, closest points, and translation gradients for spline repair.
    """

    def __init__(self, environment: dict[str, Any]) -> None:
        self.obstacles = [
            item for item in environment.get("obstacles", [])
            if item.get("collision", False) and item.get("role") != "floor"
        ]
        self._coal_checker = None
        try:
            from coal_collision import (
                CoalCollisionChecker,
                StaticCollisionObject,
            )

            static_objects = tuple(
                StaticCollisionObject.box(
                    str(item.get("id", f"obstacle_{index}")),
                    item["size_xyz"],
                    item["pose"]["position"],
                    item["pose"]["quaternion_wxyz"],
                )
                for index, item in enumerate(self.obstacles)
            )
            self._coal_checker = CoalCollisionChecker.from_urdf(
                URDF_PATH,
                static_objects,
                safety_margin=ROBOT.safety_margin,
            )
        except ModuleNotFoundError:
            # The Web UI can still run under a plain system Python.  Exact
            # closest-point repair is then explicitly reported as unavailable.
            self._coal_checker = None

    @property
    def backend_name(self) -> str:
        return (
            "coal_full_urdf_nearest_point"
            if self._coal_checker is not None
            else "urdf_primitive_obb_conservative"
        )

    @property
    def supports_clearance_gradient(self) -> bool:
        return self._coal_checker is not None

    def is_collision_free(self, position: Any, quaternion: Any) -> bool:
        if not attitude_is_within_flight_limits(quaternion):
            return False
        if self._coal_checker is not None:
            return self._coal_checker.is_collision_free(position, quaternion)
        pose = [
            *np.asarray(position, dtype=np.float64).tolist(),
            *np.asarray(quaternion, dtype=np.float64).tolist(),
        ]
        return pose_is_free(pose, self.obstacles)

    def clearance(self, positions: Any, quaternions: Any) -> np.ndarray:
        if self._coal_checker is not None:
            return self._coal_checker.clearance(positions, quaternions)
        position_array = np.asarray(positions, dtype=np.float64)
        quaternion_array = np.asarray(quaternions, dtype=np.float64)
        flat_positions = position_array.reshape(-1, 3)
        flat_quaternions = quaternion_array.reshape(-1, 4)
        values = np.asarray([
            0.10 if self.is_collision_free(position, quaternion) else -1.0
            for position, quaternion in zip(flat_positions, flat_quaternions)
        ], dtype=np.float64)
        return values.reshape(position_array.shape[:-1])

    def clearance_with_position_gradients(
        self, positions: Any, quaternions: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._coal_checker is None:
            raise RuntimeError("COAL closest-point queries are unavailable")
        return self._coal_checker.clearance_with_position_gradients(
            positions, quaternions
        )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)


def _maximum_abs_roll_pitch_degrees(states: np.ndarray) -> tuple[float, float]:
    angles = np.asarray([
        quaternion_roll_pitch_degrees(quaternion)
        for quaternion in np.asarray(states, dtype=np.float64)[:, 3:7]
    ], dtype=np.float64)
    maxima = np.max(np.abs(angles), axis=0)
    return float(maxima[0]), float(maxima[1])


def _classify_route_mode(
    states: np.ndarray,
    templates: list[tuple[str, np.ndarray]],
) -> tuple[str | None, float | None]:
    """Classify a free OMPL result by its nearest certified route signature."""
    if not templates:
        return None, None
    descriptor = _resample_positions(states)
    distances = [
        float(np.sqrt(np.mean(np.sum(
            (descriptor - _resample_positions(template)) ** 2, axis=1,
        ))))
        for _, template in templates
    ]
    index = int(np.argmin(distances))
    return templates[index][0], distances[index]


def _resample_positions(states: np.ndarray, count: int = 128) -> np.ndarray:
    distance = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1))
    ))
    if distance[-1] <= 1e-9:
        return np.repeat(states[:1, :3], count, axis=0)
    parameters = distance / distance[-1]
    targets = np.linspace(0.0, 1.0, count)
    return np.column_stack([
        np.interp(targets, parameters, states[:, axis]) for axis in range(3)
    ])


def _path_length(states: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1).sum())


def _path_naturalness_metrics(states: np.ndarray) -> dict[str, float]:
    """Measure global backtracking and local hairpin efficiency."""

    positions = _resample_positions(np.asarray(states, dtype=np.float64), 256)
    delta = np.diff(positions, axis=0)
    segment_length = np.linalg.norm(delta, axis=1)
    chord = positions[-1] - positions[0]
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm <= 1.0e-9:
        backtracking = 0.0
    else:
        forward_delta = delta @ (chord / chord_norm)
        backtracking = float(np.sum(np.maximum(0.0, -forward_delta)))
    window = 16
    window_arc = np.convolve(
        segment_length, np.ones(window), mode="valid"
    )
    window_chord = np.linalg.norm(
        positions[window:] - positions[:-window], axis=1
    )
    efficiency = np.divide(
        window_chord,
        window_arc,
        out=np.ones_like(window_chord),
        where=window_arc > 1.0e-9,
    )
    return {
        "longitudinal_backtracking_m": backtracking,
        "minimum_local_chord_efficiency": float(np.min(efficiency)),
    }


def _orientation_length_deg(states: np.ndarray) -> float:
    first, second = states[:-1, 3:7], states[1:, 3:7]
    dots = np.clip(np.abs(np.sum(first * second, axis=1)), 0.0, 1.0)
    return float(np.degrees(2 * np.arccos(dots)).sum())


def _planning_failure_stage(error: Exception) -> tuple[str, tuple[str, ...]]:
    """Classify how far an attempt progressed before the planner raised.

    ``MultiWaypointOMPLPlanner`` currently wraps the raw OMPL solve and the
    global spline in one call.  Its stable error messages nevertheless let us
    distinguish a search failure from a path that was found and subsequently
    rejected.  Keeping this translation here also prevents the Web UI from
    treating final expert yield as the raw RRTConnect success rate.
    """
    message = str(error).lower()
    if "did not find an exact path" in message:
        return "rrt_no_exact_solution", ()
    if "post-planning collision check" in message:
        return "raw_path_collision_check", ("rrt_exact_solution",)
    if "b-spline collided" in message:
        return (
            "bspline_collision",
            ("rrt_exact_solution", "raw_path_valid"),
        )
    if (
        "b-spline left the planning workspace" in message
        or "b-spline left the position bounds" in message
    ):
        return (
            "bspline_workspace_bounds",
            ("rrt_exact_solution", "raw_path_valid"),
        )
    if (
        "smoothing failed collision or workspace validation" in message
        or "global b-spline stitching failed" in message
    ):
        return (
            "bspline_validation_other",
            ("rrt_exact_solution", "raw_path_valid"),
        )
    if "start pose" in message or "goal pose" in message:
        return "invalid_endpoint", ()
    return "planner_or_pipeline_error", ()


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _direct_states(start: Any, goal: Any, count: int) -> np.ndarray:
    values = []
    first_q, second_q = start.quaternion.copy(), goal.quaternion.copy()
    if float(first_q @ second_q) < 0:
        second_q *= -1
    for alpha in np.linspace(0.0, 1.0, count):
        position = (1 - alpha) * start.position + alpha * goal.position
        quaternion = _normalize_quaternion((1 - alpha) * first_q + alpha * second_q)
        values.append([*position, *quaternion])
    return np.asarray(values, dtype=np.float64)


def generate_expert_trajectories(
    environment: dict[str, Any], *, count: int = 3, seed: int = 1,
    solve_time: float = 0.45, diversity_threshold_m: float = 0.08,
    planning_mode: str = "guided_regions",
    collision_checker: ConservativeURDFCollisionChecker | None = None,
    maximum_attempts: int | None = None,
) -> dict[str, Any]:
    if not 1 <= int(count) <= 8:
        raise ValueError("expert count must be in [1, 8]")
    if not 0.1 <= float(solve_time) <= 5.0:
        raise ValueError("solve_time must be in [0.1, 5.0] seconds per segment")
    if planning_mode == "multistage":
        # Backward-compatible API alias for scenes exported by the previous UI.
        planning_mode = "guided_regions"
    if planning_mode not in {"pure_rrtconnect", "guided_regions"}:
        raise ValueError("planning_mode must be guided_regions or pure_rrtconnect")
    certificate = environment.get("feasibility_certificate", {})
    base_route = np.asarray(certificate.get("route_poses", []), dtype=np.float64)
    if base_route.ndim != 2 or base_route.shape[1:] != (7,) or len(base_route) < 2:
        raise ValueError("scene has no valid feasibility route; regenerate the scene first")

    from multi_waypoint_planner import MultiWaypointOMPLPlanner
    from ompl_se3_planner import OMPLSE3Planner, PlannedSE3Path, SE3Pose

    class InteractiveExpertPlanner(OMPLSE3Planner):
        """Accept a dense collision-certified direct segment before sampling."""

        def plan(self, start: Any, goal: Any, **kwargs: Any) -> Any:
            minimum = int(kwargs.get("minimum_waypoints", 42))
            resolution = float(kwargs.get("interpolation_resolution", 0.07))
            distance = float(np.linalg.norm(goal.position - start.position))
            count = max(minimum, math.ceil(distance / resolution) + 1)
            direct = _direct_states(start, goal, count)
            if np.all(self.clearance(direct[:, :3], direct[:, 3:7]) > 0.0):
                return PlannedSE3Path(
                    states=direct, planning_time_s=0.0, raw_state_count=2,
                    path_length_m=_path_length(direct),
                    rotation_length_rad=math.radians(_orientation_length_deg(direct)),
                    planner_name="OMPL certified direct motion connection",
                    sampling_strategy="direct_motion_collision_check",
                    regional_sampling_probability=(
                        self.regional_sampling_probability
                    ),
                )
            return super().plan(start, goal, **kwargs)

    pairs = environment.get("precheck_pairs", [])
    if not pairs:
        raise ValueError("scene has no start/goal task pair")
    task_endpoints = np.asarray([
        pairs[0]["start_pose"], pairs[0]["goal_pose"],
    ], dtype=np.float64)
    if task_endpoints.shape != (2, 7):
        raise ValueError("scene start/goal task pair must contain two SE(3) poses")
    topology_templates = environment.get("expert_route_templates", [])
    validated_templates: list[tuple[str, np.ndarray]] = []
    for template in topology_templates:
        route = np.asarray(template.get("route_poses", []), dtype=np.float64)
        if route.ndim != 2 or route.shape[1:] != (7,) or len(route) < 3:
            raise ValueError("expert route-mode template must contain an SE(3) route")
        validated_templates.append((str(template.get("id", "topology")), route))

    raw_guides = environment.get("expert_planning_guides", [])
    if not raw_guides:
        legacy_routes = validated_templates or [("primary", base_route)]
        raw_guides = [{
            "id": topology_id,
            "policy": "fixed_waypoints_required",
            "sampled_waypoint_regions": [],
            "fixed_waypoints": route.tolist(),
            "terminal_perturbation_allowed": False,
        } for topology_id, route in legacy_routes]
    planning_guides: list[dict[str, Any]] = []
    for raw_guide in raw_guides:
        fixed = np.asarray(raw_guide.get("fixed_waypoints", []), dtype=np.float64)
        if fixed.ndim != 2 or fixed.shape[1:] != (7,) or len(fixed) < 2:
            raise ValueError("expert planning guide has invalid fixed_waypoints")
        policy = str(raw_guide.get("policy", "fixed_waypoints_required"))
        if policy not in {
            "region_biased_sampling",
            "soft_region_bias_then_fixed_fallback",
            "fixed_waypoints_required",
        }:
            raise ValueError(f"unsupported expert planning guide policy {policy!r}")
        planning_guides.append({
            **raw_guide,
            "id": str(raw_guide.get("id", "primary")),
            "policy": policy,
            "fixed_waypoints_array": fixed,
        })
    guide_lookup = {guide["id"]: guide for guide in planning_guides}

    bounds = environment["sampling_space"]["position_bounds"]
    bounds_min = np.asarray(bounds["min"], dtype=np.float64)
    bounds_max = np.asarray(bounds["max"], dtype=np.float64)
    checker = collision_checker or ConservativeURDFCollisionChecker(environment)
    if not all(checker.is_collision_free(pose[:3], pose[3:7]) for pose in base_route):
        raise ValueError("scene feasibility route is stale or colliding")

    rng = random.Random(f"experts/{seed}/{environment.get('environment_id', 'scene')}")
    accepted: list[dict[str, Any]] = []
    accepted_descriptors: list[np.ndarray] = []
    accepted_paths: list[np.ndarray] = []
    failures: list[str] = []
    pipeline_counts = {
        "attempted": 0,
        "rrt_exact_solution": 0,
        "raw_path_valid": 0,
        "bspline_valid": 0,
        "attitude_valid": 0,
        "diversity_valid": 0,
        "accepted": 0,
        "accepted_outside_target_guide": 0,
    }
    rejection_reason_counts: dict[str, int] = {}
    attempt_diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    default_maximum_attempts = (
        max(6, int(count) * 3)
        if planning_mode == "pure_rrtconnect"
        else max(12, int(count) * 6)
    )
    if maximum_attempts is None:
        maximum_attempts = default_maximum_attempts
    else:
        maximum_attempts = int(maximum_attempts)
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")
    stage_attempt_counts: dict[str, int] = {}
    guide_attempt_counts = {guide["id"]: 0 for guide in planning_guides}
    guide_accept_counts = {guide["id"]: 0 for guide in planning_guides}
    guide_region_failure_counts = {guide["id"]: 0 for guide in planning_guides}
    regional_bias_escalation_after = 4
    guide_region_enabled = {
        guide["id"]: guide["policy"] in {
            "region_biased_sampling",
            "soft_region_bias_then_fixed_fallback",
        }
        for guide in planning_guides
    }
    stopped_on_unique_capacity = False
    for attempt in range(maximum_attempts):
        trajectory_index = len(accepted)
        if trajectory_index >= count:
            break
        eligible_guides = planning_guides
        if planning_mode == "guided_regions":
            eligible_guides = [
                guide for guide in planning_guides
                if guide_region_enabled[guide["id"]]
                or guide_accept_counts[guide["id"]] == 0
            ]
            if not eligible_guides:
                stopped_on_unique_capacity = True
                break
        planner_seed = rng.randrange(1, 2**31)
        planner_range = (
            rng.uniform(0.65, 1.15)
            if planning_mode == "pure_rrtconnect"
            else rng.uniform(0.24, 0.44)
        )
        topology_class = None
        classified_topology_class = None
        route_mode_target = None
        route_mode_template_rms = None
        route_mode_target_matched = None
        generation_stage = "pure_rrtconnect"
        certificate_role = "not_used"
        hard_waypoint_indices = None
        proposal_anchor_count = 0
        sampling_regions: list[dict[str, Any]] = []
        regional_sampling_probability = 0.0
        if planning_mode == "pure_rrtconnect":
            # Deliberately expose raw RRTConnect behaviour: no gate waypoint,
            # no route region, no accepted-path perturbation and no template.
            backbone = task_endpoints.copy()
            waypoints_array = backbone.copy()
            waypoint_strategy = "pure_rrtconnect_start_goal_only"
        else:
            guide = min(
                eligible_guides,
                key=lambda item: (
                    guide_accept_counts[item["id"]],
                    guide_attempt_counts[item["id"]],
                    item["id"],
                ),
            )
            route_mode_target = guide["id"]
            topology_class = route_mode_target
            guide_attempt_counts[route_mode_target] += 1
            backbone = guide["fixed_waypoints_array"]
            use_fixed = (
                guide["policy"] == "fixed_waypoints_required"
                or not guide_region_enabled[route_mode_target]
            )
            guide_regions = list(guide.get("sampled_waypoint_regions", []))
            if not use_fixed and guide_regions:
                generation_stage = "region_biased_global"
                waypoints_array = task_endpoints.copy()
                sampling_regions = guide_regions
                base_regional_probability = float(
                    guide.get(
                        "regional_state_sampling_probability",
                        guide.get("regional_proposal_probability", 0.70),
                    )
                )
                regional_sampling_probability = min(
                    0.90,
                    base_regional_probability
                    + 0.05 * max(
                        0,
                        guide_region_failure_counts[route_mode_target]
                        - regional_bias_escalation_after + 1,
                    ),
                )
                waypoint_strategy = (
                    f"single_query_region_biased_ompl:{route_mode_target}"
                )
            else:
                if (
                    not use_fixed
                    and guide["policy"] != "fixed_waypoints_required"
                ):
                    guide_region_failure_counts[route_mode_target] += 1
                    guide_region_enabled[route_mode_target] = False
                generation_stage = "fixed_waypoint_fallback"
                certificate_role = (
                    "strict_fixed_waypoints"
                    if guide["policy"] == "fixed_waypoints_required"
                    else "region_bias_failure_fallback"
                )
                waypoints_array = backbone.copy()
                waypoint_strategy = f"scene_fixed_waypoints:{route_mode_target}"
        stage_attempt_counts[generation_stage] = (
            stage_attempt_counts.get(generation_stage, 0) + 1
        )
        pipeline_counts["attempted"] += 1
        attempt_started = time.perf_counter()
        planner_type = (
            OMPLSE3Planner
            if planning_mode == "pure_rrtconnect"
            else InteractiveExpertPlanner
        )
        planner = planner_type(
            bounds_min, bounds_max, vehicle_radius=0.0, safety_margin=0.0,
            collision_checker=checker, validity_resolution=0.004,
            planner_range=planner_range, seed=planner_seed,
            sampling_regions=sampling_regions,
            regional_sampling_probability=regional_sampling_probability,
        )
        waypoint_planner = MultiWaypointOMPLPlanner(planner)
        waypoints = [SE3Pose(pose[:3], pose[3:7]) for pose in waypoints_array]
        region_biased = generation_stage == "region_biased_global"
        effective_solve_time = solve_time
        if region_biased and route_mode_target is not None:
            escalation_level = (
                guide_region_failure_counts[route_mode_target]
                // regional_bias_escalation_after
            )
            effective_solve_time = min(
                5.0,
                solve_time * (1.0 + 0.25 * min(2, escalation_level)),
            )
        use_clearance_repair = (
            checker.supports_clearance_gradient
            and region_biased
        )
        try:
            plan = waypoint_planner.plan(
                waypoints, hard_waypoint_indices=hard_waypoint_indices,
                solve_time_per_segment=effective_solve_time,
                interpolation_resolution=0.07,
                minimum_states_per_segment=42,
                knot_stride=(12 if region_biased else 14),
                spline_samples=256, spline_method="constrained-smoothing",
                orientation_metric_weight=0.35,
                smoothing_degree=5,
                smoothing_guide_weight=(1.2 if region_biased else 0.7),
                smoothing_position_acceleration_weight=4.0e-8,
                smoothing_position_jerk_weight=5.0e-12,
                smoothing_orientation_acceleration_weight=8.0e-9,
                smoothing_orientation_jerk_weight=8.0e-13,
                smoothing_max_attempts=4,
                clearance_repair=use_clearance_repair,
                clearance_repair_margin=0.018,
                clearance_repair_samples=128,
                clearance_repair_iterations=18,
                clearance_repair_step_m=0.055,
                clearance_repair_trust_weight=0.08,
                clearance_repair_max_displacement_m=0.30,
            )
        except (RuntimeError, ValueError, ModuleNotFoundError) as error:
            if (
                planning_mode == "guided_regions"
                and route_mode_target is not None
                and generation_stage == "region_biased_global"
            ):
                guide_region_failure_counts[route_mode_target] += 1
            rejection_reason, completed_stages = _planning_failure_stage(error)
            for completed_stage in completed_stages:
                pipeline_counts[completed_stage] += 1
            _increment(rejection_reason_counts, rejection_reason)
            failures.append(f"attempt {attempt + 1}: {type(error).__name__}: {error}")
            attempt_diagnostics.append({
                "attempt": attempt + 1,
                "planner_seed": planner_seed,
                "planner_range_m": planner_range,
                "outcome": rejection_reason,
                "detail": str(error),
                "wall_time_s": time.perf_counter() - attempt_started,
            })
            continue
        pipeline_counts["rrt_exact_solution"] += 1
        pipeline_counts["raw_path_valid"] += 1
        pipeline_counts["bspline_valid"] += 1
        raw = plan.raw_states
        smooth = plan.spline_path.states
        naturalness = _path_naturalness_metrics(smooth)
        raw_roll, raw_pitch = _maximum_abs_roll_pitch_degrees(raw)
        smooth_roll, smooth_pitch = _maximum_abs_roll_pitch_degrees(smooth)
        if (
            smooth_roll > MAX_FLIGHT_ROLL_DEG + 1e-6
            or smooth_pitch > MAX_FLIGHT_PITCH_DEG + 1e-6
        ):
            _increment(rejection_reason_counts, "attitude_limit")
            failures.append(
                f"attempt {attempt + 1}: smoothed attitude exceeded limits "
                f"(roll={smooth_roll:.2f}, pitch={smooth_pitch:.2f} deg)"
            )
            attempt_diagnostics.append({
                "attempt": attempt + 1,
                "planner_seed": planner_seed,
                "planner_range_m": planner_range,
                "outcome": "attitude_limit",
                "wall_time_s": time.perf_counter() - attempt_started,
                "maximum_abs_roll_deg": smooth_roll,
                "maximum_abs_pitch_deg": smooth_pitch,
            })
            continue
        pipeline_counts["attitude_valid"] += 1
        if generation_stage == "region_biased_global" and (
            plan.maximum_curvature_per_m > 8.0
            or naturalness["longitudinal_backtracking_m"] > 0.25
            or naturalness["minimum_local_chord_efficiency"] < 0.65
        ):
            guide_region_failure_counts[route_mode_target] += 1
            _increment(rejection_reason_counts, "unnatural_geometry")
            failures.append(
                f"attempt {attempt + 1}: unnatural curve "
                f"(curvature={plan.maximum_curvature_per_m:.3f} 1/m, "
                f"backtracking={naturalness['longitudinal_backtracking_m']:.3f} m, "
                f"local_efficiency={naturalness['minimum_local_chord_efficiency']:.3f})"
            )
            attempt_diagnostics.append({
                "attempt": attempt + 1,
                "planner_seed": planner_seed,
                "planner_range_m": planner_range,
                "outcome": "unnatural_geometry",
                "wall_time_s": time.perf_counter() - attempt_started,
                "maximum_curvature_per_m": plan.maximum_curvature_per_m,
                **naturalness,
            })
            continue
        if validated_templates:
            classified_topology_class, route_mode_template_rms = _classify_route_mode(
                smooth, validated_templates,
            )
            topology_class = classified_topology_class
        if (
            generation_stage == "region_biased_global"
            and route_mode_target is not None
            and topology_class is not None
            and topology_class != route_mode_target
        ):
            # A route guide is a sampling proposal, not a closed-world
            # topological constraint. Keep a valid route discovered through
            # another corridor, account it under its realized proxy class,
            # and only strengthen future sampling for the missed target.
            route_mode_target_matched = False
            guide_region_failure_counts[route_mode_target] += 1
        elif route_mode_target is not None and topology_class is not None:
            route_mode_target_matched = topology_class == route_mode_target
        if generation_stage == "global_exploration" and topology_class is not None:
            accepted_classes = {
                expert["topology_class"] for expert in accepted
                if expert["topology_class"] is not None
            }
            target_unique_count = min(count, len(planning_guides))
            if (
                topology_class in accepted_classes
                and len(accepted_classes) < target_unique_count
            ):
                _increment(rejection_reason_counts, "topology_class_repeat")
                failures.append(
                    f"attempt {attempt + 1}: global exploration repeated "
                    f"{topology_class!r} before route-mode coverage"
                )
                attempt_diagnostics.append({
                    "attempt": attempt + 1,
                    "planner_seed": planner_seed,
                    "planner_range_m": planner_range,
                    "outcome": "topology_class_repeat",
                    "wall_time_s": time.perf_counter() - attempt_started,
                    "topology_class": topology_class,
                })
                continue
        descriptor = _resample_positions(smooth)
        comparable_indices = [
            index for index, expert in enumerate(accepted)
            if planning_mode == "pure_rrtconnect"
            or not validated_templates
            or expert["topology_class"] == topology_class
        ]
        distances = [
            float(np.sqrt(np.mean(np.sum(
                (descriptor - accepted_descriptors[index]) ** 2, axis=1,
            ))))
            for index in comparable_indices
        ]
        nearest = min(distances, default=float("inf"))
        attitude_distances = []
        length_differences = []
        for index in comparable_indices:
            reference = accepted_paths[index]
            dots = np.clip(np.abs(np.sum(
                smooth[:, 3:7] * reference[:, 3:7], axis=1,
            )), 0.0, 1.0)
            attitude_distances.append(float(np.degrees(np.sqrt(np.mean(
                np.square(2.0 * np.arccos(dots))
            )))))
            length_differences.append(abs(
                _path_length(smooth) - _path_length(reference)
            ))
        nearest_attitude = min(attitude_distances, default=float("inf"))
        nearest_length_difference = min(length_differences, default=float("inf"))
        relaxed = False
        if (
            planning_mode == "guided_regions"
            and route_mode_target is not None
        ):
            quality_guide_id = (
                topology_class if topology_class in guide_lookup
                else route_mode_target
            )
            reference_length = _path_length(
                guide_lookup[quality_guide_id]["fixed_waypoints_array"]
            )
            smooth_length = _path_length(smooth)
            if smooth_length > reference_length * 1.30 + 0.25:
                _increment(rejection_reason_counts, "excessive_detour")
                failures.append(
                    f"attempt {attempt + 1}: path length {smooth_length:.3f} m "
                    f"exceeded route-quality bound for {route_mode_target}"
                )
                attempt_diagnostics.append({
                    "attempt": attempt + 1,
                    "planner_seed": planner_seed,
                    "planner_range_m": planner_range,
                    "outcome": "excessive_detour",
                    "wall_time_s": time.perf_counter() - attempt_started,
                    "path_length_m": smooth_length,
                    "reference_length_m": reference_length,
                })
                continue
        exact_duplicate = (
            bool(distances)
            and nearest < 0.015
            and nearest_attitude < 1.5
            and nearest_length_difference < 0.02
        )
        if planning_mode == "guided_regions" and exact_duplicate:
            _increment(rejection_reason_counts, "redundant_duplicate")
            failures.append(
                f"attempt {attempt + 1}: redundant same-route expert "
                f"(position={nearest:.3f} m, attitude={nearest_attitude:.2f} deg)"
            )
            attempt_diagnostics.append({
                "attempt": attempt + 1,
                "planner_seed": planner_seed,
                "planner_range_m": planner_range,
                "outcome": "redundant_duplicate",
                "wall_time_s": time.perf_counter() - attempt_started,
                "nearest_position_diversity_m": nearest,
                "nearest_attitude_diversity_deg": nearest_attitude,
            })
            continue
        if planning_mode == "pure_rrtconnect" and distances and nearest < diversity_threshold_m:
            if attempt < maximum_attempts - max(2, count - trajectory_index):
                _increment(rejection_reason_counts, "position_diversity")
                failures.append(
                    f"attempt {attempt + 1}: diversity {nearest:.3f} m < {diversity_threshold_m:.3f} m"
                )
                attempt_diagnostics.append({
                    "attempt": attempt + 1,
                    "planner_seed": planner_seed,
                    "planner_range_m": planner_range,
                    "outcome": "position_diversity",
                    "wall_time_s": time.perf_counter() - attempt_started,
                    "nearest_position_diversity_m": nearest,
                })
                continue
            relaxed = True
        pipeline_counts["diversity_valid"] += 1
        pipeline_counts["accepted"] += 1
        if route_mode_target_matched is False:
            pipeline_counts["accepted_outside_target_guide"] += 1
        accepted.append({
            "trajectory_id": f"expert_{trajectory_index:02d}",
            "planner_attempt": attempt + 1,
            "planner_seed": planner_seed,
            "planner_range_m": planner_range,
            "solve_time_budget_s": effective_solve_time,
            "topology_class": topology_class,
            "classified_topology_class": classified_topology_class,
            "route_mode_target": route_mode_target,
            "route_mode_target_matched": route_mode_target_matched,
            "accepted_outside_target_guide": route_mode_target_matched is False,
            "route_mode_template_rms_m": route_mode_template_rms,
            "expert_generation_stage": generation_stage,
            "waypoints": waypoints_array.tolist(),
            "waypoint_strategy": waypoint_strategy,
            "proposal_anchor_count": proposal_anchor_count,
            "proposal_anchors_are_final_path_constraints": False,
            "uses_sampled_anchor_waypoints": False,
            "feasibility_certificate_used_as_global_guide": False,
            "sampled_certificate_recovery_via": (
                certificate_role == "failure_recovery_proposal_only"
            ),
            "certificate_role": certificate_role,
            "ompl_path": raw.tolist(),
            "bspline_path": smooth.tolist(),
            "metrics": {
                "planning_time_s": plan.spline_path.planning_time_s,
                "ompl_state_count": len(raw),
                "bspline_state_count": len(smooth),
                "ompl_length_m": _path_length(raw),
                "bspline_length_m": _path_length(smooth),
                "bspline_rotation_deg": _orientation_length_deg(smooth),
                "ompl_maximum_abs_roll_deg": raw_roll,
                "ompl_maximum_abs_pitch_deg": raw_pitch,
                "bspline_maximum_abs_roll_deg": smooth_roll,
                "bspline_maximum_abs_pitch_deg": smooth_pitch,
                "minimum_conservative_clearance_m": plan.minimum_clearance_m,
                "nearest_position_diversity_m": None if not distances else nearest,
                "nearest_attitude_diversity_deg": (
                    None if not attitude_distances else nearest_attitude
                ),
                "diversity_threshold_m": (
                    diversity_threshold_m
                    if planning_mode == "pure_rrtconnect" else None
                ),
                "diversity_relaxed": relaxed,
                "spline_method": plan.spline_method,
                "control_point_count": plan.control_point_count,
                "maximum_curvature_per_m": plan.maximum_curvature_per_m,
                **naturalness,
                "guide_position_rms_m": plan.guide_position_rms_m,
                "guide_attitude_rms_deg": math.degrees(plan.guide_attitude_rms_rad),
                "ompl_sampling_strategy": (
                    plan.segment_paths[0].sampling_strategy
                ),
                "regional_sampling_probability": (
                    plan.segment_paths[0].regional_sampling_probability
                ),
                "regional_sample_count": sum(
                    segment.regional_sample_count
                    for segment in plan.segment_paths
                ),
                "uniform_sample_count": sum(
                    segment.uniform_sample_count
                    for segment in plan.segment_paths
                ),
                "rejected_regional_sample_count": sum(
                    segment.rejected_regional_sample_count
                    for segment in plan.segment_paths
                ),
                "state_sampler_allocation_count": sum(
                    segment.state_sampler_allocation_count
                    for segment in plan.segment_paths
                ),
                "clearance_repair_enabled": plan.clearance_repair.enabled,
                "clearance_repair_attempted": plan.clearance_repair.attempted,
                "clearance_repair_succeeded": plan.clearance_repair.succeeded,
                "clearance_repair_target_reached": (
                    plan.clearance_repair.target_reached
                ),
                "clearance_repair_iterations": plan.clearance_repair.iterations,
                "clearance_repair_initial_minimum_m": (
                    plan.clearance_repair.initial_minimum_clearance_m
                    if np.isfinite(
                        plan.clearance_repair.initial_minimum_clearance_m
                    ) else None
                ),
                "clearance_repair_final_minimum_m": (
                    plan.clearance_repair.final_minimum_clearance_m
                    if np.isfinite(
                        plan.clearance_repair.final_minimum_clearance_m
                    ) else None
                ),
                "clearance_repair_maximum_control_displacement_m": (
                    plan.clearance_repair.maximum_control_displacement_m
                ),
            },
        })
        accepted_descriptors.append(descriptor)
        accepted_paths.append(smooth)
        accepted_guide_id = (
            topology_class if topology_class in guide_accept_counts
            else route_mode_target
        )
        if accepted_guide_id is not None:
            guide_accept_counts[accepted_guide_id] += 1
        attempt_diagnostics.append({
            "attempt": attempt + 1,
            "planner_seed": planner_seed,
            "planner_range_m": planner_range,
            "outcome": "accepted",
            "wall_time_s": time.perf_counter() - attempt_started,
            "topology_class": topology_class,
            "route_mode_target": route_mode_target,
            "route_mode_target_matched": route_mode_target_matched,
            "generation_stage": generation_stage,
            "diversity_relaxed": relaxed,
        })
    if not accepted and planning_mode != "pure_rrtconnect":
        detail = failures[-1] if failures else "no candidate returned"
        raise RuntimeError(f"OMPL expert generation failed: {detail}")
    return {
        "schema_version": "scene_expert_trajectories_v001",
        "scene_id": environment.get("environment_id"),
        "same_start_goal": True,
        "start_pose": task_endpoints[0].tolist(),
        "goal_pose": task_endpoints[-1].tolist(),
        "requested_count": int(count),
        "accepted_count": len(accepted),
        "collision_backend": checker.backend_name,
        "expert_generation_seed": int(seed),
        "planning_mode": planning_mode,
        "pipeline": (
            "pure_OMPL_RRTConnect_start_goal_then_global_SE3_BSpline"
            if planning_mode == "pure_rrtconnect"
            else "single_query_OMPL_RRTConnect_cpp_adaptive_region_global_state_sampling_then_strict_orientation_waypoints_and_global_SE3_BSpline"
        ),
        "feasibility_certificate_used_as_global_guide": False,
        "topology_constraint": (
            "classification_only_no_guidance"
            if planning_mode == "pure_rrtconnect"
            else "scene_authored_route_mode_coverage"
        ),
        "available_topology_classes": [
            guide["id"] for guide in planning_guides
        ],
        "accepted_outside_target_guide_count": sum(
            expert["accepted_outside_target_guide"] for expert in accepted
        ),
        "recovery_expert_count": sum(
            expert["certificate_role"] == "region_bias_failure_fallback"
            for expert in accepted
        ),
        "strict_fixed_expert_count": sum(
            expert["certificate_role"] == "strict_fixed_waypoints"
            for expert in accepted
        ),
        "generation_exhausted_reason": (
            "no_additional_nonredundant_guided_variants"
            if stopped_on_unique_capacity else None
        ),
        "guide_region_enabled_at_completion": guide_region_enabled,
        "generation_stage_counts": {
            stage: sum(
                expert["expert_generation_stage"] == stage
                for expert in accepted
            )
            for stage in (
                "pure_rrtconnect", "global_exploration",
                "soft_channel_proposal", "region_biased_global",
                "fixed_waypoint_fallback",
            )
        },
        "global_exploration_probability": None,
        "regional_proposal_probability": None,
        "uniform_state_sampling_probability": (
            None if planning_mode == "pure_rrtconnect" else 0.30
        ),
        "regional_state_sampling_probability": (
            None if planning_mode == "pure_rrtconnect" else 0.70
        ),
        "soft_anchor_policy": (
            None if planning_mode == "pure_rrtconnect"
            else "disabled_single_query_state_sampling_bias"
        ),
        "ordinary_guidance_uses_waypoint_constraints": False,
        "fixed_fallback_after_soft_failures": None,
        "fixed_fallback_after_region_failures": None,
        "regional_bias_escalation_after_failures": (
            regional_bias_escalation_after
        ),
        "maximum_regional_state_sampling_probability": 0.90,
        "clearance_repair": {
            "backend_available": checker.supports_clearance_gradient,
            "method": (
                "coal_nearest_point_position_gradient"
                if checker.supports_clearance_gradient else None
            ),
            "enabled_stages": [
                "region_biased_global",
            ],
            "accepted_attempted_count": sum(
                expert["metrics"]["clearance_repair_attempted"]
                for expert in accepted
            ),
            "accepted_succeeded_count": sum(
                expert["metrics"]["clearance_repair_succeeded"]
                and expert["metrics"]["clearance_repair_attempted"]
                for expert in accepted
            ),
            "target_clearance_m": 0.018,
            "position_only": True,
            "hard_waypoints_preserved": True,
            "maximum_control_displacement_m": 0.30,
        },
        "generation_stage_attempt_counts": stage_attempt_counts,
        "guide_attempt_counts": guide_attempt_counts,
        "guide_accept_counts": guide_accept_counts,
        "guide_region_failure_counts": guide_region_failure_counts,
        "diversity_policy": (
            "legacy_global_position_rms_for_diagnostic_mode"
            if planning_mode == "pure_rrtconnect"
            else "global_discovery_plus_soft_route_coverage_quality_and_duplicate_filter"
        ),
        "acceptance_pipeline": {
            "counts": pipeline_counts,
            "rates": {
                "rrt_exact_per_attempt": (
                    pipeline_counts["rrt_exact_solution"]
                    / max(1, pipeline_counts["attempted"])
                ),
                "raw_valid_per_rrt_exact": (
                    pipeline_counts["raw_path_valid"]
                    / max(1, pipeline_counts["rrt_exact_solution"])
                ),
                "bspline_valid_per_raw_valid": (
                    pipeline_counts["bspline_valid"]
                    / max(1, pipeline_counts["raw_path_valid"])
                ),
                "final_accepted_per_attempt": (
                    pipeline_counts["accepted"]
                    / max(1, pipeline_counts["attempted"])
                ),
            },
            "rejection_reason_counts": rejection_reason_counts,
            "attempts": attempt_diagnostics,
        },
        "experts": accepted,
        "recent_failures": failures[-12:],
        "total_wall_time_s": time.perf_counter() - started,
        "robot_safety_margin_m": ROBOT.safety_margin,
        "flight_attitude_limits_deg": {
            "roll": [-MAX_FLIGHT_ROLL_DEG, MAX_FLIGHT_ROLL_DEG],
            "pitch": [-MAX_FLIGHT_PITCH_DEG, MAX_FLIGHT_PITCH_DEG],
        },
    }
