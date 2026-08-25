"""Explicit extension contracts for tasks that will be connected later."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import (
    JsonObject,
    TaskCapabilities,
    TaskDescriptor,
    TaskPlugin,
    TaskUnavailableError,
)


class PlannedTaskPlugin(TaskPlugin):
    def __init__(self, descriptor: TaskDescriptor) -> None:
        self.descriptor = descriptor

    def _unavailable(self) -> TaskUnavailableError:
        return TaskUnavailableError(
            f"task {self.descriptor.task_type!r} contract is reserved; "
            "connect its condition generator, exact validity checker, and "
            "expert collector before enabling it"
        )

    def generate_condition(self, request: Mapping[str, Any]) -> JsonObject:
        raise self._unavailable()

    def validate_condition(self, condition: Mapping[str, Any]) -> list[str]:
        raise self._unavailable()

    def conditioning_payload(self, condition: Mapping[str, Any]) -> JsonObject:
        raise self._unavailable()

    def collect_experts(
        self, condition: Mapping[str, Any], request: Mapping[str, Any],
    ) -> JsonObject:
        raise self._unavailable()


def planned_task_plugins() -> list[TaskPlugin]:
    common = TaskCapabilities(
        condition_generation=False,
        condition_validation=False,
        expert_collection=False,
        batch_generation=False,
        conditioning_export=False,
        obstacle_editing=False,
        trajectory_pose_format="pose9_xyz_rotation6d",
    )
    return [
        PlannedTaskPlugin(TaskDescriptor(
            task_type="inspection",
            label="Inspection / 视觉巡检",
            summary="目标可见性、视距、视角、FOV 与遮挡联合约束的专家采集",
            status="planned_adapter",
            condition_schema="inspection_task_condition_v001",
            trajectory_schema="inspection_expert_trajectories_v001",
            capabilities=common,
            required_condition_fields=(
                "environment", "target_geometry", "sensor_extrinsics",
                "visibility_constraints", "start_goal",
            ),
            extension_notes=(
                "Adapter should reuse inspection_v2_collector exact validity.",
                "Conditioning must keep environment and target/sensor tokens separate.",
            ),
        )),
        PlannedTaskPlugin(TaskDescriptor(
            task_type="surface",
            label="Surface / 表面作业",
            summary="曲面参数域、工具外参、接触阶段与碰撞约束的专家采集",
            status="planned_adapter",
            condition_schema="surface_task_condition_v001",
            trajectory_schema="surface_expert_trajectories_v001",
            capabilities=common,
            required_condition_fields=(
                "environment", "surface_geometry", "surface_chart",
                "tool_extrinsics", "contact_constraints", "start_goal",
            ),
            extension_notes=(
                "Adapter should reuse surface_v2_collector intrinsic planning.",
                "Intrinsic states and lifted pose9 trajectories must both be retained.",
            ),
        )),
    ]
