"""Serializable campaign configuration shared by the CLI and workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from obstacle_scene_builder import FAMILY_BUILDERS


@dataclass(frozen=True)
class BatchConfig:
    schema_version: str = "expert_collection_campaign_config_v001"
    task_type: str = "free_flight"
    dataset_id: str = "free_flight_pilot_v001"
    base_seed: int = 20260825
    environment_count: int = 100
    paths_per_environment: int = 40
    nominal_conditions_per_environment: int = 8
    experts_per_condition: int = 5
    maximum_conditions_per_environment: int = 48
    workers: int = 12
    solve_time_s: float = 0.20
    maximum_planner_attempts: int = 12
    diversity_threshold_m: float = 0.08
    planning_mode: str = "guided_regions"
    obstacle_count_min: int = 6
    obstacle_count_max: int = 18
    size_min_m: float = 0.35
    size_max_m: float = 0.90
    global_yaw_min_deg: float = -35.0
    global_yaw_max_deg: float = 35.0
    translation_max_m: float = 0.30
    training_path_points: int = 128
    families: list[str] = field(default_factory=lambda: list(FAMILY_BUILDERS))

    def validate(self) -> None:
        if self.task_type != "free_flight":
            raise ValueError("this first campaign adapter supports free_flight only")
        if self.environment_count < 1 or self.paths_per_environment < 1:
            raise ValueError("environment and path targets must be positive")
        if not 1 <= self.experts_per_condition <= 8:
            raise ValueError("experts_per_condition must be in [1, 8]")
        if self.nominal_conditions_per_environment < 1:
            raise ValueError("nominal condition count must be positive")
        if self.maximum_conditions_per_environment < self.nominal_conditions_per_environment:
            raise ValueError("maximum conditions must be >= nominal conditions")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if not 0.1 <= self.solve_time_s <= 5.0:
            raise ValueError("solve_time_s must be in [0.1, 5.0]")
        if self.maximum_planner_attempts < 1:
            raise ValueError("maximum_planner_attempts must be positive")
        if self.obstacle_count_min < 1 or self.obstacle_count_max > 32:
            raise ValueError("obstacle count range must stay within [1, 32]")
        if self.obstacle_count_min > self.obstacle_count_max:
            raise ValueError("invalid obstacle count range")
        unknown = sorted(set(self.families) - set(FAMILY_BUILDERS))
        if unknown:
            raise ValueError(f"unknown scene families: {unknown}")
        if not self.families:
            raise ValueError("families must not be empty")
        if self.training_path_points < 2:
            raise ValueError("training_path_points must be at least two")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BatchConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in known})

    def dataset_signature(self) -> dict[str, Any]:
        """Fields that cannot change when resuming an existing campaign."""
        value = self.to_dict()
        value.pop("workers", None)
        return value
