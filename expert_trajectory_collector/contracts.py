"""Stable contracts between the collector core and task implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


JsonObject = dict[str, Any]


class CollectorError(RuntimeError):
    """Base class for user-facing collector failures."""


class UnknownTaskError(CollectorError):
    """Raised when a request names a task that is not registered."""


class TaskUnavailableError(CollectorError):
    """Raised when a declared future task has no collector plugin yet."""


@dataclass(frozen=True)
class TaskCapabilities:
    """Operations exposed by one task plugin and its current UI."""

    condition_generation: bool
    condition_validation: bool
    expert_collection: bool
    batch_generation: bool = False
    conditioning_export: bool = False
    obstacle_editing: bool = False
    trajectory_pose_format: str = "pose7_wxyz"


@dataclass(frozen=True)
class TaskDescriptor:
    """Serializable task discovery record returned by ``GET /api/tasks``."""

    task_type: str
    label: str
    summary: str
    status: str
    condition_schema: str
    trajectory_schema: str
    capabilities: TaskCapabilities
    required_condition_fields: tuple[str, ...] = ()
    extension_notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> JsonObject:
        result = asdict(self)
        result["available"] = self.available
        return result


class TaskPlugin(ABC):
    """Task boundary consumed by the task-agnostic collector service.

    A *condition* is intentionally more general than a free-flight scene.  It
    may be an obstacle map, an inspection target plus sensor extrinsics, or a
    surface chart plus contact/tool constraints.  Plugins are responsible for
    their own condition and trajectory schemas; the core only wraps them in a
    common collection record.
    """

    descriptor: TaskDescriptor

    def ensure_available(self) -> None:
        if not self.descriptor.available:
            raise TaskUnavailableError(
                f"task {self.descriptor.task_type!r} is registered as "
                f"{self.descriptor.status!r} but has no active collector plugin"
            )

    @abstractmethod
    def generate_condition(self, request: Mapping[str, Any]) -> JsonObject:
        """Generate one task condition from a JSON-compatible request."""

    @abstractmethod
    def validate_condition(self, condition: Mapping[str, Any]) -> list[str]:
        """Return human-readable validation issues; an empty list is valid."""

    @abstractmethod
    def conditioning_payload(self, condition: Mapping[str, Any]) -> JsonObject:
        """Encode task conditioning exactly as consumed by the model."""

    @abstractmethod
    def collect_experts(
        self, condition: Mapping[str, Any], request: Mapping[str, Any],
    ) -> JsonObject:
        """Plan, verify, and retain diverse expert trajectories."""

    def generate_batch(self, request: Mapping[str, Any]) -> JsonObject:
        raise TaskUnavailableError(
            f"task {self.descriptor.task_type!r} does not support batch generation"
        )

    def validate_expert_set(self, expert_set: Mapping[str, Any]) -> list[str]:
        trajectories = expert_set.get("experts")
        if not isinstance(trajectories, list) or not trajectories:
            return ["expert_set must contain at least one expert trajectory"]
        return []

    def present_condition(self, condition: Mapping[str, Any]) -> JsonObject:
        """Build the UI/API payload for a generated or edited condition."""

        return {
            "condition": dict(condition),
            "conditioning": self.conditioning_payload(condition),
            "validation": self.validate_condition(condition),
        }
