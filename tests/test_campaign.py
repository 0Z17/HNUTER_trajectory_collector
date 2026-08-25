from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from expert_trajectory_collector.campaign.config import BatchConfig
from expert_trajectory_collector.campaign.encoding import (
    pose7_to_pose9,
    resample_pose7_path,
)
from expert_trajectory_collector.campaign.free_flight import sample_condition
from expert_trajectory_collector.campaign.state import aggregate_status, set_paused
from obstacle_scene_builder import generate_scene, validate_scene


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


def test_pause_marker_and_aggregate_status(tmp_path: Path) -> None:
    config = BatchConfig(environment_count=2, paths_per_environment=3, workers=1)
    (tmp_path / "campaign_config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8",
    )
    set_paused(tmp_path, True)
    assert aggregate_status(tmp_path)["state"] == "paused"
    set_paused(tmp_path, False)
    assert aggregate_status(tmp_path)["paused"] is False
