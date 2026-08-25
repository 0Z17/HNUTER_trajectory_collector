from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from obstacle_scene_builder import (
    FAMILY_BUILDERS,
    FAMILY_MINIMUMS,
    GUARANTEED_MULTI_ROUTE_FAMILIES,
    MAX_FLIGHT_PITCH_DEG,
    MAX_FLIGHT_ROLL_DEG,
    MAX_OBSTACLES,
    ROBOT,
    generate_batch,
    generate_scene,
    obstacles_overlap,
    obstacle_tokens,
    pose_is_free,
    passage_limits,
    quaternion_matrix,
    quaternion_roll_pitch_degrees,
    route_is_free,
    validate_scene,
    _endpoint_regions,
    _sample_region,
)


@pytest.mark.parametrize("family", FAMILY_BUILDERS)
def test_every_family_is_deterministic_valid_and_honours_count(family: str) -> None:
    count = max(7, FAMILY_MINIMUMS[family])
    parameters = {
        "family": family,
        "seed": 9281,
        "obstacle_count": count,
        "gap_width": sum(passage_limits(family)) / 2 if family == "orientation_sensitive_passage" else 1.7,
        "global_yaw_deg": 12,
    }
    first = generate_scene(parameters)
    second = generate_scene(parameters)
    assert first == second
    assert len([item for item in first["obstacles"] if item["role"] != "floor"]) == count
    assert validate_scene(first) == []


def test_exported_tokens_match_training_encoder_exactly() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("se3_diffusion")
    from se3_diffusion import _obstacle_tokens

    scene = generate_scene({
        "family": "sparse_obb_clutter", "seed": 17, "obstacle_count": 9,
    })
    web_tokens, web_mask = obstacle_tokens(scene)
    training_tokens, training_mask = _obstacle_tokens(scene)
    np.testing.assert_array_equal(np.asarray(web_tokens, dtype=np.float32), training_tokens)
    np.testing.assert_array_equal(np.asarray(web_mask, dtype=bool), training_mask)
    assert np.asarray(web_tokens).shape == (MAX_OBSTACLES, 10)
    assert sum(web_mask) == 9


def test_floor_is_not_encoded_and_invalid_edits_are_reported() -> None:
    scene = generate_scene({"family": "frame_doorway", "obstacle_count": 5})
    _, mask = obstacle_tokens(scene)
    assert sum(mask) == 5

    broken = copy.deepcopy(scene)
    broken["obstacles"][1]["pose"]["position"][0] = 4.5
    broken["obstacles"][2]["pose"]["quaternion_wxyz"] = [2, 0, 0, 0]
    issues = validate_scene(broken)
    assert any("固定场景边界" in issue for issue in issues)
    assert any("未归一化" in issue for issue in issues)


def test_template_minimum_prevents_losing_semantic_structure() -> None:
    with pytest.raises(ValueError, match="needs at least 6 obstacles"):
        generate_scene({
            "family": "orientation_sensitive_passage", "obstacle_count": 3,
        })


def test_batch_randomizes_all_families_and_only_returns_valid_scenes() -> None:
    bank = generate_batch({
        "variants_per_family": 2, "base_seed": 314, "obstacle_count_max": 7,
    })
    assert bank["scene_count"] == 2 * len(FAMILY_BUILDERS)
    assert {scene["generation_family"] for scene in bank["scenes"]} == set(FAMILY_BUILDERS)
    assert all(not validate_scene(scene) for scene in bank["scenes"])
    assert bank == generate_batch({
        "variants_per_family": 2, "base_seed": 314, "obstacle_count_max": 7,
    })


def test_urdf_envelope_drives_collision_clearance() -> None:
    assert ROBOT.collision_names == (
        "base_collision", "l_leg_collision", "l_rotor_collision",
        "r_leg_collision", "r_rotor_collision", "tail_leg_collision",
        "tail_rotoer_collision",
    )
    np.testing.assert_allclose(ROBOT.size, [1.218555, 1.29494, 0.546803], atol=1e-6)
    assert ROBOT.safety_margin == pytest.approx(0.129494)


@pytest.mark.parametrize("family", FAMILY_BUILDERS)
def test_range_sampling_is_diverse_non_overlapping_and_route_feasible(family: str) -> None:
    scenes = []
    for seed in (101, 202, 303):
        attitude = family == "orientation_sensitive_passage"
        attitude_limits = passage_limits(family) if attitude else (1.53, 1.90)
        scenes.append(generate_scene({
            "family": family, "seed": seed, "sample_ranges": True,
            "obstacle_count_min": max(5, FAMILY_MINIMUMS[family]),
            "obstacle_count_max": 10, "size_min": 0.35, "size_max": 1.05,
            "gap_width_min": attitude_limits[0],
            "gap_width_max": attitude_limits[1],
        }))
    signatures = {
        tuple(
            (item["id"], *item["pose"]["position"], *item["size_xyz"])
            for item in scene["obstacles"] if item["role"] != "floor"
        )
        for scene in scenes
    }
    assert len(signatures) == len(scenes)
    for scene in scenes:
        active = [item for item in scene["obstacles"] if item["role"] != "floor"]
        for index, first in enumerate(active):
            for second in active[index + 1:]:
                if first.get("assembly_group") == second.get("assembly_group") and first.get("assembly_group"):
                    continue
                assert not obstacles_overlap(first, second)
        for obstacle in active:
            rotation = quaternion_matrix(obstacle["pose"]["quaternion_wxyz"])
            world_half = np.abs(rotation) @ (
                np.asarray(obstacle["size_xyz"], dtype=np.float64) / 2
            )
            bottom = float(obstacle["pose"]["position"][2] - world_half[2])
            assert bottom <= 0.03 or obstacle.get("physical_support")
        certificate = scene["feasibility_certificate"]
        assert route_is_free(certificate["route_poses"], active)
        route_positions = np.asarray(certificate["route_poses"])[:, :3]
        position_bounds = scene["sampling_space"]["position_bounds"]
        assert np.all(route_positions >= np.asarray(position_bounds["min"]))
        assert np.all(route_positions <= np.asarray(position_bounds["max"]))
        for endpoint in (certificate["route_poses"][0], certificate["route_poses"][-1]):
            rotation = quaternion_matrix(endpoint[3:7])
            tilt_degrees = np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
            assert tilt_degrees >= 3.0
        assert scene["robot_reference"]["flight_attitude_limits_deg"] == {
            "roll": [-MAX_FLIGHT_ROLL_DEG, MAX_FLIGHT_ROLL_DEG],
            "pitch": [-MAX_FLIGHT_PITCH_DEG, MAX_FLIGHT_PITCH_DEG],
        }
        for pose in certificate["route_poses"]:
            roll_degrees, pitch_degrees = quaternion_roll_pitch_degrees(pose[3:7])
            assert abs(roll_degrees) <= MAX_FLIGHT_ROLL_DEG + 1e-6
            assert abs(pitch_degrees) <= MAX_FLIGHT_PITCH_DEG + 1e-6
        assert scene["task_sampling"]["regions_are_not_fixed_points"] is True
        guides = scene["expert_planning_guides"]
        assert [guide["id"] for guide in guides]
        assert all(guide["terminal_perturbation_allowed"] is False for guide in guides)
        if family == "orientation_sensitive_passage":
            assert len(guides) == 1
            assert guides[0]["policy"] == "fixed_waypoints_required"
            assert guides[0]["sampled_waypoint_regions"] == []
            assert len(guides[0]["fixed_waypoints"]) == 4
            assert guides[0]["strict_mode_constraint"] == (
                "orientation_coupled_aperture"
            )
        else:
            assert all(
                guide["policy"] == "region_biased_sampling"
                and guide["uniform_state_sampling_probability"] == 0.30
                and guide["regional_state_sampling_probability"] == 0.70
                and guide["region_samples_are_waypoints"] is False
                and (
                    2 <= len(guide["sampled_waypoint_regions"]) <= 4
                    if family in {
                        "staggered_corridor", "mixed_industrial",
                        "wall_protrusion_bracket",
                    }
                    else len(guide["sampled_waypoint_regions"]) == 2
                )
                for guide in guides
            )
        for name, pose in (
            ("start_region", certificate["route_poses"][0]),
            ("goal_region", certificate["route_poses"][-1]),
        ):
            region = scene["task_sampling"][name]
            local = quaternion_matrix(region["quaternion_wxyz"]).T @ (
                np.asarray(pose[:3]) - np.asarray(region["center"])
            )
            assert np.all(np.abs(local) <= np.asarray(region["size_xyz"]) / 2 + 1e-5)


@pytest.mark.parametrize("family", sorted(GUARANTEED_MULTI_ROUTE_FAMILIES))
def test_guaranteed_multi_route_families_protect_shared_endpoint_modes(
    family: str,
) -> None:
    scene = generate_scene({
        "family": family, "seed": 202,
        "obstacle_count": max(7, FAMILY_MINIMUMS[family]),
        "gap_width": 1.7,
    })
    templates = scene["expert_route_templates"]
    assert scene["route_mode_policy"] == "guaranteed_multi"
    assert scene["verified_route_mode_count"] == len(templates)
    assert len(templates) >= 2
    active = [item for item in scene["obstacles"] if item["role"] != "floor"]
    assert all(route_is_free(template["route_poses"], active) for template in templates)
    endpoints = [
        np.asarray([template["route_poses"][0], template["route_poses"][-1]])
        for template in templates
    ]
    assert all(np.allclose(endpoint, endpoints[0]) for endpoint in endpoints[1:])


def test_orientation_gate_has_visible_ground_support_columns() -> None:
    scene = generate_scene({
        "family": "orientation_sensitive_passage", "seed": 876,
        "obstacle_count": 6,
    })
    supports = [
        item for item in scene["obstacles"]
        if item.get("role") == "structural_support"
    ]
    assert {item["id"] for item in supports} == {
        "slot_low_support", "slot_high_support",
    }
    assert all(item.get("physical_support") == "ground" for item in supports)


def test_doorway_keeps_an_outer_bypass_near_the_old_display_cutoff() -> None:
    scene = generate_scene({
        "family": "frame_doorway", "seed": 115,
        "sample_ranges": True, "obstacle_count_min": 6,
        "obstacle_count_max": 10,
    })
    assert scene["verified_route_mode_count"] >= 2
    assert {item["id"] for item in scene["expert_route_templates"]} >= {
        "through",
    }


def test_central_block_uses_geometry_adaptive_oriented_route_slabs() -> None:
    scene = generate_scene({
        # Inspect the complete core route bank without functional fillers,
        # which are now allowed to retire some certificates dynamically.
        "family": "central_block", "seed": 202, "obstacle_count": 1,
    })
    block = next(
        item for item in scene["obstacles"] if item["id"] == "central_block"
    )
    guides = {item["id"]: item for item in scene["expert_planning_guides"]}
    above = guides["above"]["sampled_waypoint_regions"]
    left = guides["left"]["sampled_waypoint_regions"]
    right = guides["right"]["sampled_waypoint_regions"]
    assert all(
        region["mode_constraint"] == "vertical_bypass_slab"
        and region["size_xyz"][1] >= block["size_xyz"][1]
        and region["size_xyz"][2] > 0.35
        for region in above
    )
    assert all(
        region["mode_constraint"] == "lateral_bypass_slab"
        and region["size_xyz"][1] > 0.60
        and region["size_xyz"][2] > 1.0
        for region in [*left, *right]
    )
    assert all(
        region["orientation_sampling"] == "bounded_reference_rpy_jitter"
        and len(region["quaternion_wxyz"]) == 4
        for guide in guides.values()
        for region in guide["sampled_waypoint_regions"]
    )


def test_remaining_families_use_structure_aware_planning_regions() -> None:
    doorway = generate_scene({
        "family": "frame_doorway", "seed": 202,
        "obstacle_count": FAMILY_MINIMUMS["frame_doorway"],
        "gap_width": 1.7,
    })
    doorway_guides = {
        guide["id"]: guide for guide in doorway["expert_planning_guides"]
    }
    assert {
        region["mode_constraint"]
        for region in doorway_guides["through"]["sampled_waypoint_regions"]
    } == {"doorway_aperture_cross_section"}
    assert all(
        region["mode_constraint"] == "vertical_bypass_slab"
        and region["size_xyz"][1] > 2.0
        for region in doorway_guides["above"]["sampled_waypoint_regions"]
    )
    side_guides = [
        guide for topology_id, guide in doorway_guides.items()
        if topology_id in {"left", "right"}
    ]
    assert side_guides
    assert all(
        region["mode_constraint"] == "lateral_bypass_slab"
        for guide in side_guides
        for region in guide["sampled_waypoint_regions"]
    )

    for family, expected_constraint in (
        ("narrow_passage", "narrow_aperture_cross_section"),
    ):
        scene = generate_scene({
            "family": family, "seed": 202,
            "obstacle_count": FAMILY_MINIMUMS[family],
            "gap_width": 1.7,
        })
        regions = scene["expert_planning_guides"][0][
            "sampled_waypoint_regions"
        ]
        assert len(regions) == 2
        assert all(
            region["mode_constraint"] == expected_constraint
            and region["size_xyz"][1] <= 0.20
            and region["size_xyz"][2] > 2.0
            and region["coherent_sampling_strength"][1:] == [1.0, 1.0]
            for region in regions
        )

    bracket_scene = generate_scene({
        "family": "wall_protrusion_bracket", "seed": 202,
        "obstacle_count": 7, "gap_width": 1.7,
    })
    bracket_regions = bracket_scene["expert_planning_guides"][0][
        "sampled_waypoint_regions"
    ]
    assert len(bracket_regions) == 4
    assert [region["mode_constraint"] for region in bracket_regions] == [
        "bracket_dogleg_clearance", "bracket_dogleg_clearance",
        "bracket_aperture_cross_section", "bracket_aperture_cross_section",
    ]

    for family, expected_constraint in (
        ("staggered_corridor", "staggered_clearance_chain"),
    ):
        scene = generate_scene({
            "family": family, "seed": 202,
            "obstacle_count": FAMILY_MINIMUMS[family], "gap_width": 1.7,
        })
        for guide in scene["expert_planning_guides"]:
            regions = guide["sampled_waypoint_regions"]
            assert 2 <= len(regions) <= 4
            assert all(
                region["mode_constraint"] == expected_constraint
                and region["size_xyz"][1] > 0.58
                and region["size_xyz"][2] > 1.0
                and region["coherent_sampling_strength"][1] < 1.0
                for region in regions
            )

    industrial = generate_scene({
        "family": "mixed_industrial", "seed": 202,
        "obstacle_count": 7, "gap_width": 1.7,
    })
    industrial_regions = industrial["expert_planning_guides"][0][
        "sampled_waypoint_regions"
    ]
    assert len(industrial_regions) == 4
    assert all(
        region["mode_constraint"] == "industrial_candidate_soft_channel"
        and region["coherent_sampling_strength"][1] < 1.0
        for region in industrial_regions
    )
    assert industrial["verified_route_mode_count"] >= 2
    assert {
        guide["id"] for guide in industrial["expert_planning_guides"]
    } >= {"winding_under", "free_space_alternative"}


def test_bracket_and_industrial_core_assemblies_change_the_task_corridor() -> None:
    bracket = generate_scene({
        "family": "wall_protrusion_bracket", "seed": 202,
        "obstacle_count": 7, "gap_width": 1.7,
    })
    bracket_route = bracket["feasibility_certificate"]["route_poses"]
    bracket_obstacles = [
        item for item in bracket["obstacles"] if item["role"] != "floor"
    ]
    straight_portal = [
        bracket_route[0], bracket_route[3], bracket_route[5],
        bracket_route[-1],
    ]
    without_protrusion = [
        item for item in bracket_obstacles
        if item["id"].startswith("bracket_wall")
        or item["role"] == "secondary_obstacle"
    ]
    assert not route_is_free(straight_portal, bracket_obstacles)
    assert route_is_free(straight_portal, without_protrusion)
    assert all(
        item.get("route_relevant") is True
        for item in bracket_obstacles
        if item["role"] != "secondary_obstacle"
    )

    industrial = generate_scene({
        "family": "mixed_industrial", "seed": 202,
        "obstacle_count": FAMILY_MINIMUMS["mixed_industrial"],
        "gap_width": 1.7,
    })
    industrial_route = industrial["feasibility_certificate"]["route_poses"]
    industrial_obstacles = [
        item for item in industrial["obstacles"] if item["role"] != "floor"
    ]
    assert not route_is_free(
        [industrial_route[0], industrial_route[-1]], industrial_obstacles,
    )
    route_y = np.asarray(industrial_route)[:, 1]
    assert np.min(route_y[1:-1]) <= -1.0
    assert np.max(route_y[1:-1]) >= 1.0
    assert all(item.get("route_relevant") is True for item in industrial_obstacles)


def test_multi_homotopy_scene_preserves_vertical_and_lateral_route_families() -> None:
    scene = generate_scene({
        "family": "multi_homotopy", "seed": 202,
        "sample_ranges": True, "obstacle_count_min": 5,
        "obstacle_count_max": 9,
    })
    templates = scene["expert_route_templates"]
    template_ids = {template["id"] for template in templates}
    summary = scene["obstacle_function_summary"]
    assert summary["initial_route_mode_count"] == 4
    assert 2 <= len(template_ids) <= 4
    assert template_ids & {"above", "below"}
    assert template_ids & {"left", "right"}
    active = [item for item in scene["obstacles"] if item["role"] != "floor"]
    assert all(route_is_free(template["route_poses"], active) for template in templates)
    routes = {
        template["id"]: np.asarray(template["route_poses"])
        for template in templates
    }
    if {"above", "below"}.issubset(routes):
        assert np.mean(routes["above"][1:-1, 2]) > np.mean(
            routes["below"][1:-1, 2]
        )
    if {"left", "right"}.issubset(routes):
        left_y = np.mean(routes["left"][1:-1, 1])
        right_y = np.mean(routes["right"][1:-1, 1])
        assert left_y > right_y
    supports = [
        item for item in active if item.get("role") == "structural_support"
    ]
    assert {item["id"] for item in supports} == {
        "separator_support_left", "separator_support_right",
    }


def test_terminal_regions_overlap_but_span_low_and_high_altitudes() -> None:
    deltas = []
    for seed in range(100):
        rng = random.Random(seed)
        start_region, goal_region = _endpoint_regions(rng)
        start_low = start_region["center"][2] - start_region["size_xyz"][2] / 2
        start_high = start_region["center"][2] + start_region["size_xyz"][2] / 2
        goal_low = goal_region["center"][2] - goal_region["size_xyz"][2] / 2
        goal_high = goal_region["center"][2] + goal_region["size_xyz"][2] / 2
        assert start_region["size_xyz"][2] >= 1.85
        assert goal_region["size_xyz"][2] >= 1.85
        assert max(start_low, goal_low) < min(start_high, goal_high)
        deltas.append(abs(
            _sample_region(start_region, rng)[2]
            - _sample_region(goal_region, rng)[2]
        ))
    assert min(deltas) < 0.05
    assert max(deltas) > 1.30


def test_variation_props_are_mostly_flight_relevant_height() -> None:
    scene = generate_scene({
        "family": "sparse_obb_clutter", "seed": 9281,
        "obstacle_count": 7,
    })
    props = [
        item for item in scene["obstacles"]
        if item["role"] == "secondary_obstacle"
    ]
    tall_or_medium = [item for item in props if item["size_xyz"][2] >= 1.45]
    assert len(tall_or_medium) >= int(0.50 * len(props))


def test_variation_props_report_task_space_function_and_preserve_route_floor() -> None:
    scene = generate_scene({
        "family": "central_block", "seed": 2026,
        "obstacle_count": 12,
    })
    active = [
        item for item in scene["obstacles"] if item["role"] != "floor"
    ]
    props = [
        item for item in active if item["role"] == "secondary_obstacle"
    ]
    summary = scene["obstacle_function_summary"]
    assert summary["generated_prop_count"] == len(props)
    assert summary["effective_prop_ratio"] >= 0.65
    assert summary["route_selector_count"] >= 1
    assert summary["surviving_route_mode_count"] >= 2
    assert summary["surviving_route_mode_count"] >= summary[
        "minimum_surviving_route_mode_count"
    ]
    assert summary["blocked_route_modes"]
    assert all(
        route_is_free(template["route_poses"], active)
        for template in scene["expert_route_templates"]
    )
    for prop in props:
        evidence = prop["influence_evidence"]
        if prop["functional_role"] == "route_selector":
            assert evidence["blocked_certificate_modes"]
        elif prop["functional_role"] == "clearance_shaper":
            assert evidence["clearance_shell_modes"]
            assert evidence["clearance_shell_m"] == pytest.approx(0.42)
        else:
            assert prop["functional_role"] == "distractor"
            assert not evidence["blocked_certificate_modes"]
            assert not evidence["clearance_shell_modes"]


def test_orientation_gate_rejects_upright_but_certifies_rolled_pose() -> None:
    limits = passage_limits("orientation_sensitive_passage")
    scene = generate_scene({
        "family": "orientation_sensitive_passage", "seed": 876,
        "sample_ranges": True, "obstacle_count_min": 4, "obstacle_count_max": 7,
        "gap_width_min": limits[0], "gap_width_max": limits[1],
    })
    active = [item for item in scene["obstacles"] if item["role"] != "floor"]
    gate = next(item for item in active if item["id"] == "slot_ceiling_lip")
    rolled_pose = max(
        scene["feasibility_certificate"]["route_poses"],
        key=lambda pose: abs(pose[4]) + abs(pose[5]),
    )
    yaw = np.radians(scene["generation_parameters"]["global_yaw_deg"])
    upright_q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    rotated_center = quaternion_matrix(upright_q) @ np.asarray(ROBOT.center)
    portal = gate["aperture_center_world_xyz"]
    upright = [
        portal[0] - rotated_center[0], portal[1] - rotated_center[1],
        portal[2] - rotated_center[2], *upright_q,
    ]
    assert not pose_is_free(upright, active)
    assert pose_is_free(rolled_pose, active)
    assert scene["feasibility_certificate"]["requires_attitude_change"] is True
    roll_degrees = abs(float(gate["required_roll_deg"]))
    assert 25.0 <= roll_degrees <= MAX_FLIGHT_ROLL_DEG
