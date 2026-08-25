from __future__ import annotations

from typing import Any, Mapping

import pytest

from expert_trajectory_collector import (
    ExpertCollectorService,
    TaskCapabilities,
    TaskDescriptor,
    TaskPlugin,
    TaskRegistry,
    TaskUnavailableError,
    create_default_registry,
)
from expert_trajectory_collector.cli import parse_args
from expert_trajectory_collector.service import COLLECTION_RECORD_SCHEMA
from expert_trajectory_collector.web import ExpertCollectorHandler, make_handler


class SyntheticTaskPlugin(TaskPlugin):
    descriptor = TaskDescriptor(
        task_type="synthetic",
        label="Synthetic",
        summary="test task",
        status="available",
        condition_schema="synthetic_condition_v001",
        trajectory_schema="synthetic_trajectories_v001",
        capabilities=TaskCapabilities(True, True, True),
    )

    def generate_condition(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return {"value": int(request.get("value", 1))}

    def validate_condition(self, condition: Mapping[str, Any]) -> list[str]:
        return [] if "value" in condition else ["missing value"]

    def conditioning_payload(self, condition: Mapping[str, Any]) -> dict[str, Any]:
        return {"scalar": int(condition["value"])}

    def collect_experts(
        self, condition: Mapping[str, Any], request: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {"experts": [{"trajectory_id": "synthetic_00"}]}


def test_default_catalog_declares_current_and_future_tasks() -> None:
    service = ExpertCollectorService(create_default_registry())
    catalog = service.list_tasks()
    tasks = {item["task_type"]: item for item in catalog["tasks"]}
    assert catalog["default_task"] == "free_flight"
    assert tasks["free_flight"]["available"] is True
    assert tasks["inspection"]["available"] is False
    assert tasks["surface"]["available"] is False
    assert "sensor_extrinsics" in tasks["inspection"]["required_condition_fields"]
    assert "surface_chart" in tasks["surface"]["required_condition_fields"]


def test_free_flight_adapter_preserves_legacy_payload_and_adds_contract() -> None:
    service = ExpertCollectorService(create_default_registry())
    payload = service.generate({
        "task_type": "free_flight",
        "family": "central_block",
        "seed": 91,
        "obstacle_count": 5,
    })
    assert payload["scene"] == payload["condition"]
    assert payload["scene"]["schema_version"] == "free_flight_scene_v001"
    assert payload["conditioning"]["schema_version"] == "box_geometry_v001"
    assert payload["conditioning"]["tokens"] == payload["tokens"]
    assert payload["conditioning"]["mask"] == payload["mask"]
    assert payload["task_descriptor"]["trajectory_schema"] == (
        "scene_expert_trajectories_v001"
    )
    assert payload["validation"] == []
    # Legacy validate clients post the raw scene, without task/condition keys.
    validated = service.validate(payload["scene"])
    assert validated["scene"] == payload["scene"]
    assert validated["validation"] == []


def test_collection_record_keeps_condition_conditioning_and_expert_set() -> None:
    service = ExpertCollectorService(TaskRegistry([SyntheticTaskPlugin()]), "synthetic")
    expert_set = {"experts": [{"trajectory_id": "synthetic_00"}]}
    record = service.build_collection_record({
        "task_type": "synthetic",
        "condition": {"value": 7},
        "expert_set": expert_set,
    })
    assert record["schema_version"] == COLLECTION_RECORD_SCHEMA
    assert record["task_type"] == "synthetic"
    assert record["condition"] == {"value": 7}
    assert record["conditioning"] == {"scalar": 7}
    assert record["expert_set"] == expert_set


def test_registry_accepts_new_task_without_core_service_changes() -> None:
    service = ExpertCollectorService(TaskRegistry([SyntheticTaskPlugin()]), "synthetic")
    generated = service.generate({"value": 12})
    assert generated["condition"] == {"value": 12}
    assert service.collect({
        "condition": generated["condition"],
    })["experts"][0]["trajectory_id"] == "synthetic_00"


def test_planned_tasks_fail_explicitly_instead_of_using_free_flight() -> None:
    service = ExpertCollectorService(create_default_registry())
    with pytest.raises(TaskUnavailableError, match="planned_adapter"):
        service.generate({"task_type": "inspection"})


def test_free_flight_condition_to_expert_collection_record_integration() -> None:
    pytest.importorskip("ompl")
    service = ExpertCollectorService(create_default_registry())
    generated = service.generate({
        "family": "central_block", "seed": 119, "obstacle_count": 3,
    })
    expert_set = service.collect({
        "condition": generated["condition"],
        "count": 1, "seed": 37, "solve_time": 0.2,
        "planning_mode": "guided_regions",
    })
    record = service.build_collection_record({
        "condition": generated["condition"], "expert_set": expert_set,
    })
    assert expert_set["accepted_count"] == 1
    assert record["expert_set"]["schema_version"] == (
        "scene_expert_trajectories_v001"
    )
    assert record["condition"]["environment_id"] == expert_set["scene_id"]
    assert record["conditioning"]["schema_version"] == "box_geometry_v001"


def test_cli_and_web_transport_are_collector_owned() -> None:
    args = parse_args([
        "--task", "free_flight", "--generate", "--family", "central_block",
    ])
    assert args.task == "free_flight"
    assert args.generate is True
    service = ExpertCollectorService(TaskRegistry([SyntheticTaskPlugin()]), "synthetic")
    handler = make_handler(service)
    assert issubclass(handler, ExpertCollectorHandler)
    assert handler.service is service
