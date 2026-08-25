from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from expert_trajectory_collector.campaign.config import BatchConfig
from expert_trajectory_collector.campaign import free_flight
from expert_trajectory_collector.campaign.encoding import (
    pose7_to_pose9,
    resample_pose7_path,
)
from expert_trajectory_collector.campaign.free_flight import sample_condition
from expert_trajectory_collector.campaign.state import aggregate_status, set_paused
from obstacle_scene_builder import (
    MAX_FLIGHT_ROLL_DEG,
    SMOOTHING_ATTITUDE_GUARD_DEG,
    generate_scene,
    validate_scene,
)


def test_pose9_encoding_and_arc_resampling() -> None:
    path = np.asarray([
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0.9238795, 0, 0, 0.3826834],
    ], dtype=np.float64)
    sampled, progress = resample_pose7_path(path, 17)
    encoded = pose7_to_pose9(sampled)
    assert sampled.shape == (17, 7)
    assert encoded.shape == (17, 9)
    assert np.allclose(progress, np.linspace(0, 1, 17))
    assert np.allclose(encoded[0, 3:], [1, 0, 0, 0, 1, 0])


def test_condition_sampler_changes_endpoints_and_preserves_validity() -> None:
    scene = generate_scene({
        "family": "central_block", "seed": 31415,
        "obstacle_count": 3,
    })
    first = sample_condition(scene, 0, 10)
    second = sample_condition(scene, 1, 11)
    assert first["condition_id"] != second["condition_id"]
    assert first["obstacles"] == second["obstacles"]
    assert first["precheck_pairs"][0]["start_pose"] != second["precheck_pairs"][0]["start_pose"]
    assert validate_scene(first) == []
    assert validate_scene(second) == []


def test_condition_sampler_prunes_invalid_optional_route_modes() -> None:
    scene = generate_scene({
        "family": "central_block", "seed": 31415,
        "obstacle_count": 1,
    })
    scene["route_mode_policy"] = "optional_multi"
    blocked_id = scene["expert_planning_guides"][0]["id"]
    obstacle = next(
        item for item in scene["obstacles"] if item.get("role") != "floor"
    )
    blocked_pose = [
        *obstacle["pose"]["position"], 1.0, 0.0, 0.0, 0.0,
    ]
    scene["expert_planning_guides"][0]["fixed_waypoints"][1] = blocked_pose
    for template in scene["expert_route_templates"]:
        if template["id"] == blocked_id:
            template["route_poses"][1] = blocked_pose
    endpoints = scene["precheck_pairs"][0]
    condition = free_flight._condition_with_valid_routes(
        scene, endpoints["start_pose"], endpoints["goal_pose"],
        condition_index=0, seed=10, sampling_attempt=0,
        sampling_elapsed_s=0.01,
    )
    assert condition is not None
    validation = condition["condition_route_validation"]
    assert blocked_id in validation["dropped_mode_ids"]
    assert blocked_id not in validation["retained_mode_ids"]
    assert validation["all_environment_routes_required"] is False
    assert validate_scene(condition) == []


def test_guaranteed_multi_condition_keeps_at_least_two_modes() -> None:
    scene = generate_scene({
        "family": "central_block", "seed": 31415,
        "obstacle_count": 1,
    })
    scene["expert_planning_guides"] = scene["expert_planning_guides"][:1]
    scene["expert_route_templates"] = scene["expert_route_templates"][:1]
    endpoints = scene["precheck_pairs"][0]
    assert free_flight._condition_with_valid_routes(
        scene, endpoints["start_pose"], endpoints["goal_pose"],
        condition_index=0, seed=10, sampling_attempt=0,
        sampling_elapsed_s=0.01,
    ) is None


def test_condition_failure_circuit_breaker_stops_environment(
    tmp_path: Path, monkeypatch,
) -> None:
    config = BatchConfig(
        environment_count=1, paths_per_environment=5,
        maximum_conditions_per_environment=20,
        maximum_consecutive_condition_failures=3,
        environment_precheck_condition_count=0,
        families=["central_block"], workers=1,
    )
    environment_dir = tmp_path / "environments" / "env_000000"
    environment_dir.mkdir(parents=True)
    scene = generate_scene({
        "family": "central_block", "seed": 31415,
        "obstacle_count": 1,
    })
    (environment_dir / "environment.json").write_text(
        json.dumps(scene), encoding="utf-8",
    )

    def fail_condition(*_args, **_kwargs):
        raise ValueError(
            "could not sample a start/goal condition with the required valid route modes"
        )

    monkeypatch.setattr(free_flight, "sample_condition", fail_condition)
    progress = free_flight.collect_environment(
        config.to_dict(), str(tmp_path), 0,
    )
    assert progress["worker_state"] == "capacity_exhausted"
    assert progress["condition_count"] == 3
    assert progress["termination_reason"] == "consecutive_condition_failure_limit"
    assert progress["condition_failure_reason_counts"] == {
        "condition_sampling": 3,
    }


def test_orientation_gate_generation_reserves_smoothing_margin() -> None:
    for seed in range(10):
        scene = generate_scene({
            "family": "orientation_sensitive_passage", "seed": seed,
            "obstacle_count": 6,
        })
        required_roll = abs(float(next(
            item["required_roll_deg"] for item in scene["obstacles"]
            if "required_roll_deg" in item
        )))
        assert required_roll <= (
            MAX_FLIGHT_ROLL_DEG - SMOOTHING_ATTITUDE_GUARD_DEG
        )


def test_v001_config_resume_preserves_legacy_sampling_behavior() -> None:
    config = BatchConfig.from_dict({
        "schema_version": "expert_collection_campaign_config_v001",
        "dataset_id": "legacy",
    })
    assert config.condition_sampling_max_attempts == 160
    assert config.condition_sampling_timeout_s == 0.0
    assert config.environment_precheck_condition_count == 0
    assert config.maximum_consecutive_condition_failures == 48
    assert config.terminal_attitude_margin_deg == 0.0
    assert config.experts_per_condition_overrides == {}
    previous_v002 = BatchConfig.from_dict({
        "schema_version": "expert_collection_campaign_config_v002",
        "dataset_id": "previous_v002",
    })
    assert previous_v002.experts_per_condition_overrides == {}


def test_single_channel_families_request_one_expert_per_condition(
    tmp_path: Path, monkeypatch,
) -> None:
    requested_by_family: dict[str, list[int]] = {}

    monkeypatch.setattr(
        free_flight, "ConservativeURDFCollisionChecker",
        lambda _environment: SimpleNamespace(backend_name="test_collision"),
    )
    monkeypatch.setattr(
        free_flight, "sample_condition",
        lambda environment, *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(free_flight, "save_expert_set", lambda *_args: None)

    for family in (
        "orientation_sensitive_passage", "staggered_corridor", "central_block",
    ):
        output = tmp_path / family
        environment_dir = output / "environments" / "env_000000"
        environment_dir.mkdir(parents=True)
        (environment_dir / "environment.json").write_text(json.dumps({
            "environment_id": f"test_{family}",
            "generation_family": family,
        }), encoding="utf-8")
        requested: list[int] = []

        def generate_experts(_condition, *, count: int, **_kwargs):
            requested.append(count)
            return {
                "accepted_count": count,
                "experts": [],
                "acceptance_pipeline": {"counts": {"attempted": count}},
            }

        monkeypatch.setattr(
            free_flight, "generate_expert_trajectories", generate_experts,
        )
        config = BatchConfig(
            environment_count=1, paths_per_environment=5,
            maximum_conditions_per_environment=5,
            environment_precheck_condition_count=0,
            families=[family], workers=1,
        )
        progress = free_flight.collect_environment(
            config.to_dict(), str(output), 0,
        )
        assert progress["worker_state"] == "complete"
        requested_by_family[family] = requested

    assert requested_by_family["orientation_sensitive_passage"] == [1] * 5
    assert requested_by_family["staggered_corridor"] == [1] * 5
    assert requested_by_family["central_block"] == [5]


def test_pause_marker_and_aggregate_status(tmp_path: Path) -> None:
    config = BatchConfig(environment_count=2, paths_per_environment=3, workers=1)
    (tmp_path / "campaign_config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8",
    )
    set_paused(tmp_path, True)
    assert aggregate_status(tmp_path)["state"] == "paused"
    set_paused(tmp_path, False)
    assert aggregate_status(tmp_path)["paused"] is False


def test_aggregate_status_reports_unreachable_shortfall(tmp_path: Path) -> None:
    config = BatchConfig(environment_count=2, paths_per_environment=3, workers=1)
    (tmp_path / "campaign_config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8",
    )
    for index, (state, accepted) in enumerate((
        ("complete", 3), ("capacity_exhausted", 1),
    )):
        directory = tmp_path / "environments" / f"env_{index:06d}"
        directory.mkdir(parents=True)
        (directory / "progress.json").write_text(json.dumps({
            "worker_state": state,
            "accepted_path_count": accepted,
            "planner_attempt_count": 0,
        }), encoding="utf-8")
    status = aggregate_status(tmp_path)
    assert status["state"] == "finished_with_shortfall"
    assert status["environment_finished"] == 2
    assert status["path_shortfall"] == 2
    assert status["reachable_path_upper_bound"] == 4
    assert status["target_reachable"] is False
    assert status["eta_s"] is None


def test_concurrent_status_writes_use_independent_temporary_files(
    tmp_path: Path,
) -> None:
    config = BatchConfig(environment_count=1, paths_per_environment=1, workers=1)
    (tmp_path / "campaign_config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8",
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(
            lambda _index: aggregate_status(tmp_path), range(40),
        ))
    assert all(status["path_target"] == 1 for status in statuses)
    assert json.loads((tmp_path / "status.json").read_text())["path_target"] == 1
    assert not list(tmp_path.glob(".status.json.*.tmp"))
