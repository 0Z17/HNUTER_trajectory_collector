"""Adapter from the existing free-flight domain to collector contracts."""

from __future__ import annotations

from typing import Any, Mapping

from obstacle_scene_builder import (
    generate_batch,
    generate_scene,
    obstacle_tokens,
    validate_scene,
)

from ..contracts import JsonObject, TaskCapabilities, TaskDescriptor, TaskPlugin


class FreeFlightTaskPlugin(TaskPlugin):
    descriptor = TaskDescriptor(
        task_type="free_flight",
        label="Free-flight / 障碍绕行",
        summary="OBB 场景生成、SE(3) OMPL 专家规划与 B-spline 平滑",
        status="available",
        condition_schema="free_flight_scene_v001",
        trajectory_schema="scene_expert_trajectories_v001",
        capabilities=TaskCapabilities(
            condition_generation=True,
            condition_validation=True,
            expert_collection=True,
            batch_generation=True,
            conditioning_export=True,
            obstacle_editing=True,
            trajectory_pose_format="pose7_xyz_quaternion_wxyz",
        ),
        required_condition_fields=(
            "obstacles", "sampling_space", "precheck_pairs",
            "feasibility_certificate",
        ),
        extension_notes=(
            "Conditioning is padded 10-D OBB geometry plus a validity mask.",
            "Expert output retains both raw OMPL and smoothed B-spline paths.",
        ),
    )

    def generate_condition(self, request: Mapping[str, Any]) -> JsonObject:
        return generate_scene(dict(request))

    def validate_condition(self, condition: Mapping[str, Any]) -> list[str]:
        return validate_scene(dict(condition))

    def conditioning_payload(self, condition: Mapping[str, Any]) -> JsonObject:
        tokens, mask = obstacle_tokens(dict(condition))
        return {
            "schema_version": "box_geometry_v001",
            "feature_order": [
                "x", "y", "z", "size_x", "size_y", "size_z",
                "qw", "qx", "qy", "qz",
            ],
            "tokens": tokens,
            "mask": mask,
        }

    def present_condition(self, condition: Mapping[str, Any]) -> JsonObject:
        conditioning = self.conditioning_payload(condition)
        # Preserve the original Web/API shape while exposing the generic
        # condition terminology alongside it.
        return {
            "scene": dict(condition),
            "condition": dict(condition),
            "tokens": conditioning["tokens"],
            "mask": conditioning["mask"],
            "conditioning": conditioning,
            "validation": self.validate_condition(condition),
        }

    def generate_batch(self, request: Mapping[str, Any]) -> JsonObject:
        return generate_batch(dict(request))

    def collect_experts(
        self, condition: Mapping[str, Any], request: Mapping[str, Any],
    ) -> JsonObject:
        from obstacle_scene_experts import generate_expert_trajectories

        return generate_expert_trajectories(
            dict(condition),
            count=int(request.get("count", 3)),
            seed=int(request.get("seed", 1)),
            solve_time=float(request.get("solve_time", 0.45)),
            diversity_threshold_m=float(
                request.get("diversity_threshold_m", 0.08)
            ),
            planning_mode=str(request.get("planning_mode", "guided_regions")),
        )

    def validate_expert_set(self, expert_set: Mapping[str, Any]) -> list[str]:
        issues = super().validate_expert_set(expert_set)
        if issues:
            return issues
        if expert_set.get("schema_version") != "scene_expert_trajectories_v001":
            return ["free-flight expert_set has an unexpected schema_version"]
        for index, expert in enumerate(expert_set["experts"]):
            if not isinstance(expert, Mapping):
                return [f"expert[{index}] must be an object"]
            for field in ("ompl_path", "bspline_path"):
                path = expert.get(field)
                if not isinstance(path, list) or len(path) < 2:
                    return [f"expert[{index}].{field} must contain a trajectory"]
        return []
