"""Tests for true in-query regional sampling bias in OMPL."""

from __future__ import annotations

import numpy as np
import pytest

from ompl_se3_planner import (
    OMPLSE3Planner,
    SE3Pose,
    SphereObstacle,
)


def test_region_biased_rrtconnect_uses_cpp_sampler_without_waypoints() -> None:
    pytest.importorskip("ompl")
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    planner = OMPLSE3Planner(
        (-2.0, -2.0, -1.0),
        (2.0, 2.0, 1.0),
        obstacles=(SphereObstacle(np.zeros(3), 0.55),),
        vehicle_radius=0.10,
        safety_margin=0.05,
        validity_resolution=0.001,
        seed=11,
        sampling_regions=({
            "center_pose": [0.0, 1.0, 0.0, *identity],
            "size_xyz": [1.5, 0.8, 0.5],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "orientation_rpy_jitter_deg": [4.0, 4.0, 8.0],
        },),
        regional_sampling_probability=0.70,
    )
    start = SE3Pose(np.asarray([-1.5, 0.0, 0.0]), identity)
    goal = SE3Pose(np.asarray([1.5, 0.0, 0.0]), identity)

    path = planner.plan(
        start,
        goal,
        solve_time=1.0,
        minimum_waypoints=60,
    )

    assert path.planner_name == "OMPL RRTConnect (C++ region-biased StateSampler)"
    assert path.sampling_strategy == "cpp_state_sampler_mixture"
    assert path.regional_sampling_probability == 0.70
    assert path.state_sampler_allocation_count >= 1
    assert path.regional_sample_count > 0
    assert path.uniform_sample_count > 0
    np.testing.assert_allclose(path.states[0, :3], start.position)
    np.testing.assert_allclose(path.states[-1, :3], goal.position)
    assert np.all(
        planner.clearance(path.states[:, :3], path.states[:, 3:7]) > 0.0
    )
