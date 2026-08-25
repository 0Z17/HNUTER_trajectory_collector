"""Application service for condition generation and expert collection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from .contracts import JsonObject, TaskPlugin
from .registry import TaskRegistry


COLLECTION_RECORD_SCHEMA = "expert_trajectory_collection_record_v001"


class ExpertCollectorService:
    """Task-agnostic use cases shared by the CLI, Web UI, and future jobs."""

    def __init__(self, registry: TaskRegistry, default_task: str = "free_flight") -> None:
        if default_task not in registry:
            raise ValueError(f"default task {default_task!r} is not registered")
        self.registry = registry
        self.default_task = default_task

    def list_tasks(self) -> JsonObject:
        return {
            "schema_version": "expert_collector_task_catalog_v001",
            "default_task": self.default_task,
            "tasks": self.registry.descriptors(),
        }

    def _task_type(self, request: Mapping[str, Any]) -> str:
        return str(request.get("task_type", self.default_task))

    def _plugin(self, request: Mapping[str, Any]) -> TaskPlugin:
        plugin = self.registry.get(self._task_type(request))
        plugin.ensure_available()
        return plugin

    @staticmethod
    def _condition(request: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = request.get("condition", request.get("scene"))
        if nested is None:
            # Backward compatibility: legacy /api/validate posts the scene
            # object itself rather than wrapping it in a request envelope.
            nested = request
        if not isinstance(nested, Mapping):
            raise ValueError("request requires a JSON condition/scene object")
        return nested

    @staticmethod
    def _decorate(payload: JsonObject, plugin: TaskPlugin) -> JsonObject:
        return {
            **payload,
            "task_type": plugin.descriptor.task_type,
            "task_descriptor": plugin.descriptor.to_dict(),
        }

    def generate(self, request: Mapping[str, Any]) -> JsonObject:
        plugin = self._plugin(request)
        condition = plugin.generate_condition(request)
        return self._decorate(plugin.present_condition(condition), plugin)

    def validate(self, request: Mapping[str, Any]) -> JsonObject:
        plugin = self._plugin(request)
        return self._decorate(
            plugin.present_condition(self._condition(request)), plugin,
        )

    def batch(self, request: Mapping[str, Any]) -> JsonObject:
        plugin = self._plugin(request)
        return self._decorate(plugin.generate_batch(request), plugin)

    def collect(self, request: Mapping[str, Any]) -> JsonObject:
        plugin = self._plugin(request)
        condition = self._condition(request)
        issues = plugin.validate_condition(condition)
        if issues:
            raise ValueError(
                "cannot collect experts for an invalid condition: "
                + "; ".join(issues)
            )
        result = plugin.collect_experts(condition, request)
        return self._decorate(result, plugin)

    def build_collection_record(self, request: Mapping[str, Any]) -> JsonObject:
        plugin = self._plugin(request)
        condition = self._condition(request)
        issues = plugin.validate_condition(condition)
        if issues:
            raise ValueError(
                "cannot export an invalid condition: " + "; ".join(issues)
            )
        expert_set = request.get("expert_set", request.get("experts"))
        if not isinstance(expert_set, Mapping):
            raise ValueError("collection export requires an expert_set object")
        expert_issues = plugin.validate_expert_set(expert_set)
        if expert_issues:
            raise ValueError(
                "cannot export an invalid expert set: "
                + "; ".join(expert_issues)
            )
        return {
            "schema_version": COLLECTION_RECORD_SCHEMA,
            "task_type": plugin.descriptor.task_type,
            "task_contract": plugin.descriptor.to_dict(),
            "condition": dict(condition),
            "conditioning": plugin.conditioning_payload(condition),
            "expert_set": dict(expert_set),
            "collection_metadata": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "collector": "HNUTER Expert Trajectory Studio",
            },
        }
