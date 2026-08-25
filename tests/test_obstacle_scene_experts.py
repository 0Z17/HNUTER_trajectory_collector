from __future__ import annotations

import numpy as np
import pytest

import obstacle_scene_experts as expert_module
from obstacle_scene_builder import (
    MAX_FLIGHT_PITCH_DEG,
    MAX_FLIGHT_ROLL_DEG,
    generate_scene,
    passage_limits,
)
from obstacle_scene_experts import (
    ConservativeURDFCollisionChecker,
    generate_expert_trajectories,
)


def test_interactive_expert_pipeline_exposes_ompl_and_bspline() -> None:
    pytest.importorskip("ompl")
    scene = generate_scene({
        "family": "sparse_obb_clutter", "seed": 707,
        "sample_ranges": True, "obstacle_count_min": 5,
        "obstacle_count_max": 6,
    })
    result = generate_expert_trajectories(
        scene, count=1, seed=19, solve_time=0.2,
        planning_mode="guided_regions",
    )
    assert result["accepted_count"] == 1
    assert result["feasibility_certificate_used_as_global_guide"] is False
    expert = result["experts"][0]
    assert expert["expert_generation_stage"] in {
        "region_biased_global", "fixed_waypoint_fallback",
    }
    assert "small_local_via" not in expert["waypoint_strategy"]
    assert len(expert["waypoints"]) >= 2
    if expert["expert_generation_stage"] == "region_biased_global":
        assert expert["proposal_anchor_count"] == 0
        assert expert["proposal_anchors_are_final_path_constraints"] is False
        assert expert["uses_sampled_anchor_waypoints"] is False
        assert len(expert["waypoints"]) == 2
        assert expert["metrics"]["ompl_sampling_strategy"] in {
            "cpp_state_sampler_mixture", "direct_motion_collision_check",
        }
        if expert["metrics"]["ompl_sampling_strategy"] == "cpp_state_sampler_mixture":
            assert expert["metrics"]["regional_sample_count"] > 0
            assert expert["metrics"]["state_sampler_allocation_count"] > 0
        assert expert["metrics"]["maximum_curvature_per_m"] <= 8.0
        assert expert["metrics"]["longitudinal_backtracking_m"] <= 0.25
        assert expert["metrics"]["minimum_local_chord_efficiency"] >= 0.65
    ompl = np.asarray(expert["ompl_path"])
    bspline = np.asarray(expert["bspline_path"])
    assert ompl.shape[1] == bspline.shape[1] == 7
    assert len(bspline) == 256
    assert expert["metrics"]["spline_method"] == "waypoint-constrained smoothing"
    assert expert["metrics"]["control_point_count"] * 4 < len(ompl)
    assert expert["metrics"]["ompl_maximum_abs_roll_deg"] <= MAX_FLIGHT_ROLL_DEG
    assert expert["metrics"]["ompl_maximum_abs_pitch_deg"] <= MAX_FLIGHT_PITCH_DEG
    assert expert["metrics"]["bspline_maximum_abs_roll_deg"] <= MAX_FLIGHT_ROLL_DEG
    assert expert["metrics"]["bspline_maximum_abs_pitch_deg"] <= MAX_FLIGHT_PITCH_DEG
    assert result["flight_attitude_limits_deg"] == {
        "roll": [-MAX_FLIGHT_ROLL_DEG, MAX_FLIGHT_ROLL_DEG],
        "pitch": [-MAX_FLIGHT_PITCH_DEG, MAX_FLIGHT_PITCH_DEG],
    }
    np.testing.assert_allclose(ompl[0], bspline[0], atol=1e-7)
    np.testing.assert_allclose(ompl[-1], bspline[-1], atol=1e-7)
    checker = ConservativeURDFCollisionChecker(scene)
    assert np.all(checker.clearance(bspline[:, :3], bspline[:, 3:7]) > 0)

    refreshed = generate_expert_trajectories(
        scene, count=1, seed=20, solve_time=0.2,
        planning_mode="guided_regions",
    )
    refreshed_path = np.asarray(refreshed["experts"][0]["bspline_path"])
    position_rms = np.sqrt(np.mean(np.sum(
        np.square(bspline[:, :3] - refreshed_path[:, :3]), axis=1,
    )))
    # Repeated geometry is allowed to remain close; the planner must never add
    # an obstacle-free terminal detour merely to cross an RMS threshold.
    if position_rms <= 0.02:
        assert "small_local_via" not in expert["waypoint_strategy"]
        assert "small_local_via" not in refreshed["experts"][0]["waypoint_strategy"]


def test_multi_homotopy_scene_regions_cover_all_surviving_classes() -> None:
    pytest.importorskip("ompl")
    scene = generate_scene({
        "family": "multi_homotopy", "seed": 202,
        "sample_ranges": True, "obstacle_count_min": 5,
        "obstacle_count_max": 7,
    })
    expected_classes = [
        guide["id"] for guide in scene["expert_planning_guides"]
    ]
    assert set(expected_classes) & {"above", "below"}
    assert set(expected_classes) & {"left", "right"}
    result = generate_expert_trajectories(
        scene, count=len(expected_classes), seed=31, solve_time=0.2,
        planning_mode="guided_regions",
    )
    assert result["available_topology_classes"] == expected_classes
    assert {expert["topology_class"] for expert in result["experts"]} == set(
        expected_classes
    )
    assert all(
        expert["classified_topology_class"] == expert["topology_class"]
        for expert in result["experts"]
    )
    assert (
        result["generation_stage_counts"]["region_biased_global"]
        + result["generation_stage_counts"]["fixed_waypoint_fallback"]
        == result["accepted_count"]
    )
    assert result["global_exploration_probability"] is None
    assert result["regional_proposal_probability"] is None
    assert result["uniform_state_sampling_probability"] == 0.30
    assert result["regional_state_sampling_probability"] == 0.70
    assert result["ordinary_guidance_uses_waypoint_constraints"] is False
    assert all(
        len(expert["waypoints"]) == 2
        and expert["proposal_anchor_count"] == 0
        for expert in result["experts"]
        if expert["expert_generation_stage"] == "region_biased_global"
    )
    assert all(
        expert["metrics"]["maximum_curvature_per_m"] <= 8.0
        and expert["metrics"]["longitudinal_backtracking_m"] <= 0.25
        and expert["metrics"]["minimum_local_chord_efficiency"] >= 0.65
        for expert in result["experts"]
        if expert["expert_generation_stage"] == "region_biased_global"
    )
    assert result["acceptance_pipeline"]["rejection_reason_counts"].get(
        "position_diversity", 0,
    ) == 0


def test_pure_rrtconnect_never_uses_route_guides_or_class_backfill() -> None:
    pytest.importorskip("ompl")
    scene = generate_scene({
        "family": "multi_homotopy", "seed": 202,
        "obstacle_count": 7,
    })
    result = generate_expert_trajectories(
        scene, count=2, seed=37, solve_time=0.15,
        planning_mode="pure_rrtconnect",
    )
    assert result["planning_mode"] == "pure_rrtconnect"
    assert result["topology_constraint"] == "classification_only_no_guidance"
    assert result["generation_stage_attempt_counts"]["pure_rrtconnect"] <= 6
    assert result["generation_stage_counts"]["global_exploration"] == 0
    assert result["generation_stage_counts"]["soft_channel_proposal"] == 0
    assert result["generation_stage_counts"]["region_biased_global"] == 0
    assert result["generation_stage_counts"]["fixed_waypoint_fallback"] == 0
    pipeline = result["acceptance_pipeline"]
    counts = pipeline["counts"]
    assert counts["attempted"] == result["generation_stage_attempt_counts"]["pure_rrtconnect"]
    assert (
        counts["attempted"] >= counts["rrt_exact_solution"]
        >= counts["raw_path_valid"] >= counts["bspline_valid"]
        >= counts["attitude_valid"] >= counts["diversity_valid"]
        == counts["accepted"] == result["accepted_count"]
    )
    assert sum(pipeline["rejection_reason_counts"].values()) + counts["accepted"] == counts["attempted"]
    assert len(pipeline["attempts"]) == counts["attempted"]
    assert all(len(expert["waypoints"]) == 2 for expert in result["experts"])
    assert all(
        expert["waypoint_strategy"] == "pure_rrtconnect_start_goal_only"
        and expert["route_mode_target"] is None
        and expert["certificate_role"] == "not_used"
        for expert in result["experts"]
    )


def test_central_block_experts_follow_surviving_route_modes() -> None:
    pytest.importorskip("ompl")
    scene = generate_scene({
        "family": "central_block", "seed": 202, "obstacle_count": 7,
    })
    result = generate_expert_trajectories(
        scene, count=2, seed=31, solve_time=0.2,
        planning_mode="guided_regions",
    )
    assert result["topology_constraint"] == "scene_authored_route_mode_coverage"
    assert result["available_topology_classes"] == [
        guide["id"] for guide in scene["expert_planning_guides"]
    ]
    assert len({expert["topology_class"] for expert in result["experts"]}) == 2
    assert result["diversity_policy"] == (
        "global_discovery_plus_soft_route_coverage_quality_and_duplicate_filter"
    )
    assert all(
        expert["expert_generation_stage"] in {
            "region_biased_global", "fixed_waypoint_fallback",
        }
        for expert in result["experts"]
    )


def test_valid_route_outside_target_guide_is_accepted_and_reclassified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("ompl")
    scene = generate_scene({
        "family": "central_block", "seed": 202, "obstacle_count": 7,
    })
    guide_ids = sorted(
        guide["id"] for guide in scene["expert_planning_guides"]
    )
    # Make the selected proposal deterministic: the planner chooses the
    # lexicographically first uncovered guide, while the classifier below
    # reports the last one.  This test targets acceptance semantics, not the
    # stochastic success order of regional RRTConnect queries.
    for guide in scene["expert_planning_guides"]:
        guide["policy"] = "fixed_waypoints_required"
        guide["sampled_waypoint_regions"] = []
    realized_class = guide_ids[-1]
    monkeypatch.setattr(
        expert_module,
        "_classify_route_mode",
        lambda states, templates: (realized_class, 0.20),
    )

    result = generate_expert_trajectories(
        scene, count=1, seed=31, solve_time=0.2,
        planning_mode="guided_regions",
    )

    expert = result["experts"][0]
    assert expert["route_mode_target"] != realized_class
    assert expert["topology_class"] == realized_class
    assert expert["route_mode_target_matched"] is False
    assert expert["accepted_outside_target_guide"] is True
    assert result["accepted_outside_target_guide_count"] == 1
    assert result["guide_accept_counts"][realized_class] == 1
    assert "route_mode_mismatch" not in (
        result["acceptance_pipeline"]["rejection_reason_counts"]
    )


def test_orientation_gate_is_not_confused_by_a_tilted_terminal_pose() -> None:
    pytest.importorskip("ompl")
    lower, upper = passage_limits("orientation_sensitive_passage")
    scene = generate_scene({
        "family": "orientation_sensitive_passage", "seed": 876,
        "sample_ranges": True, "obstacle_count_min": 4,
        "obstacle_count_max": 7, "gap_width_min": lower,
        "gap_width_max": upper,
    })
    result = generate_expert_trajectories(
        scene, count=4, seed=91, solve_time=0.3,
        planning_mode="guided_regions",
    )
    assert result["accepted_count"] == 1
    expert = result["experts"][0]
    assert expert["waypoint_strategy"] == "scene_fixed_waypoints:primary"
    assert expert["expert_generation_stage"] == "fixed_waypoint_fallback"
    assert "small_local_via" not in expert["waypoint_strategy"]
    assert len(expert["waypoints"]) == 4
    assert result["strict_fixed_expert_count"] == 1
    assert result["generation_exhausted_reason"] == (
        "no_additional_nonredundant_guided_variants"
    )
    assert expert["metrics"]["bspline_maximum_abs_roll_deg"] <= MAX_FLIGHT_ROLL_DEG
    assert expert["metrics"]["bspline_maximum_abs_pitch_deg"] <= MAX_FLIGHT_PITCH_DEG


@pytest.mark.parametrize(
    "family", ["wall_protrusion_bracket", "mixed_industrial"],
)
def test_complex_families_use_bounded_soft_attempts_then_strict_fallback(
    family: str,
) -> None:
    pytest.importorskip("ompl")
    scene = generate_scene({
        "family": family, "seed": 202,
        "obstacle_count": 7, "gap_width": 1.7,
    })
    result = generate_expert_trajectories(
        scene, count=1, seed=31, solve_time=0.15,
        planning_mode="guided_regions",
    )
    assert len(result["available_topology_classes"]) >= 2
    assert result["fixed_fallback_after_soft_failures"] is None
    assert result["fixed_fallback_after_region_failures"] is None
    assert result["regional_bias_escalation_after_failures"] == 4
    assert result["accepted_count"] == 1
    assert result["generation_stage_attempt_counts"].get(
        "region_biased_global", 0,
    ) >= 1
    expert = result["experts"][0]
    if expert["expert_generation_stage"] == "region_biased_global":
        assert len(expert["waypoints"]) == 2
        assert expert["proposal_anchor_count"] == 0
        assert expert["metrics"]["regional_sample_count"] > 0
