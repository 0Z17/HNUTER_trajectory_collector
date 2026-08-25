"""Tests for multi-waypoint OMPL planning and global B-spline timing."""

from __future__ import annotations

import unittest

import numpy as np

from mppi.quaternion import (
    quaternion_error_vector,
    quaternion_from_euler,
)
from multi_waypoint_planner import (
    BSplineTimeParameterizedReference,
    InterpolatingSE3BSpline,
    MultiWaypointOMPLPlanner,
    WaypointConstrainedSmoothingSE3BSpline,
)
from ompl_se3_planner import (
    OMPLSE3Planner,
    SE3Pose,
    SphereObstacle,
)


class MultiWaypointPlannerTest(unittest.TestCase):
    def test_clearance_gradient_repair_preserves_endpoints(self) -> None:
        class RadialChecker:
            supports_clearance_gradient = True

            def is_collision_free(self, position, quaternion) -> bool:
                del quaternion
                return bool(np.linalg.norm(np.asarray(position)[:2]) > 0.65)

            def clearance(self, positions, quaternions):
                del quaternions
                return np.linalg.norm(
                    np.asarray(positions)[..., :2], axis=-1
                ) - 0.65

            def clearance_with_position_gradients(
                self, positions, quaternions,
            ):
                del quaternions
                positions = np.asarray(positions, dtype=np.float64)
                radius = np.linalg.norm(positions[..., :2], axis=-1)
                gradient = np.zeros_like(positions)
                gradient[..., :2] = positions[..., :2] / np.maximum(
                    radius[..., None], 1.0e-9
                )
                return radius - 0.65, gradient

        x = np.linspace(-1.0, 1.0, 31)
        positions = np.column_stack((
            x, 0.65 * (1.0 - np.square(x)), np.ones_like(x)
        ))
        guide = np.column_stack((
            positions,
            np.tile((1.0, 0.0, 0.0, 0.0), (len(x), 1)),
        ))
        spline = WaypointConstrainedSmoothingSE3BSpline(
            guide,
            (0, len(guide) - 1),
            control_point_stride=8,
            guide_weight=0.1,
            position_acceleration_weight=1.0e-4,
        )
        checker = RadialChecker()
        planner = MultiWaypointOMPLPlanner(OMPLSE3Planner(
            (-2.0, -2.0, 0.0),
            (2.0, 2.0, 2.0),
            collision_checker=checker,
        ))
        endpoints_before = spline.evaluate(np.asarray([0.0, 1.0]))
        quaternion_controls_before = spline.quaternion_control_points.copy()
        dense_before = spline.evaluate(np.linspace(0.0, 1.0, 500))
        self.assertLess(
            float(np.min(checker.clearance(
                dense_before[:, :3], dense_before[:, 3:7]
            ))),
            0.0,
        )

        diagnostics = planner._repair_spline_clearance(
            spline,
            enabled=True,
            margin=0.03,
            sample_count=128,
            maximum_iterations=25,
            maximum_step_m=0.06,
            trust_weight=0.03,
            maximum_displacement_m=0.40,
        )
        dense_after = spline.evaluate(np.linspace(0.0, 1.0, 1000))

        self.assertTrue(diagnostics.attempted)
        self.assertTrue(diagnostics.succeeded)
        self.assertTrue(diagnostics.target_reached)
        self.assertGreater(
            float(np.min(checker.clearance(
                dense_after[:, :3], dense_after[:, 3:7]
            ))),
            0.0,
        )
        np.testing.assert_allclose(
            spline.evaluate(np.asarray([0.0, 1.0])),
            endpoints_before,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            spline.quaternion_control_points,
            quaternion_controls_before,
            atol=0.0,
        )

    def test_soft_proposal_anchor_is_released_before_global_smoothing(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        waypoints = (
            SE3Pose(np.asarray([-1.0, 0.0, 1.0]), identity),
            SE3Pose(np.asarray([0.0, 1.0, 1.0]), identity),
            SE3Pose(np.asarray([1.0, 0.0, 1.0]), identity),
        )
        planner = OMPLSE3Planner(
            bounds_min=(-1.5, -0.5, 0.5),
            bounds_max=(1.5, 1.5, 1.5),
            planner_range=0.4,
            seed=23,
        )
        plan = MultiWaypointOMPLPlanner(planner).plan(
            waypoints,
            hard_waypoint_indices=(0, 2),
            solve_time_per_segment=0.2,
            minimum_states_per_segment=20,
            knot_stride=4,
            spline_samples=160,
            smoothing_guide_weight=0.25,
            smoothing_position_acceleration_weight=2.0e-5,
        )
        self.assertEqual(len(plan.segment_paths), 2)
        self.assertEqual(len(plan.waypoints), 2)
        np.testing.assert_allclose(
            plan.spline_path.states[[0, -1], :3],
            np.asarray([waypoints[0].position, waypoints[-1].position]),
            atol=1.0e-7,
        )
        self.assertGreater(
            float(np.min(np.linalg.norm(
                plan.spline_path.states[:, :3] - waypoints[1].position,
                axis=1,
            ))),
            1.0e-3,
        )

    def test_segment_position_bounds_are_enforced_by_rrt(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        waypoints = (
            SE3Pose(np.asarray([-1.2, -1.0, 1.0]), identity),
            SE3Pose(np.asarray([-0.8, 0.0, 1.0]), identity),
            SE3Pose(np.asarray([1.2, 1.0, 1.0]), identity),
        )
        planner = OMPLSE3Planner(
            bounds_min=(-2.0, -2.0, 0.2),
            bounds_max=(2.0, 2.0, 2.0),
            planner_range=0.35,
            seed=31,
        )
        bounds = (
            (np.asarray([-2.0, -1.2, 0.8]), np.asarray([-0.6, 0.1, 1.2])),
            (np.asarray([-0.9, -0.1, 0.8]), np.asarray([2.0, 1.2, 1.2])),
        )
        plan = MultiWaypointOMPLPlanner(planner).plan(
            waypoints,
            segment_position_bounds=bounds,
            solve_time_per_segment=0.3,
            minimum_states_per_segment=20,
            knot_stride=2,
            spline_samples=120,
        )
        for segment, (lower, upper) in zip(plan.segment_paths, bounds):
            self.assertTrue(np.all(segment.states[:, :3] >= lower - 1.0e-9))
            self.assertTrue(np.all(segment.states[:, :3] <= upper + 1.0e-9))

    def test_constrained_smoother_preserves_hard_waypoints_and_reduces_curvature(
        self,
    ) -> None:
        parameters = np.linspace(0.0, 1.0, 41)
        positions = np.column_stack(
            (
                3.0 * parameters,
                0.22 * np.sin(2.0 * np.pi * parameters)
                + 0.045 * np.sin(16.0 * np.pi * parameters),
                1.0
                + 0.15 * np.sin(np.pi * parameters)
                + 0.03 * np.sin(14.0 * np.pi * parameters),
            )
        )
        quaternions = quaternion_from_euler(
            np.column_stack(
                (
                    0.12 * np.sin(2.0 * np.pi * parameters),
                    0.10 * np.sin(3.0 * np.pi * parameters),
                    1.20 * parameters
                    + 0.04 * np.sin(14.0 * np.pi * parameters),
                )
            )
        )
        guide_states = np.concatenate(
            (positions, quaternions), axis=1
        )
        hard_indices = (0, 20, 40)
        interpolating = InterpolatingSE3BSpline(
            guide_states, parameters=parameters
        )
        smoothed = WaypointConstrainedSmoothingSE3BSpline(
            guide_states,
            hard_indices,
            parameters=parameters,
            control_point_stride=4,
        )

        hard_states = smoothed.evaluate(
            smoothed.waypoint_parameters
        )
        np.testing.assert_allclose(
            hard_states[:, :3],
            guide_states[list(hard_indices), :3],
            atol=1.0e-10,
        )
        hard_attitude_error = np.linalg.norm(
            quaternion_error_vector(
                hard_states[:, 3:7],
                guide_states[list(hard_indices), 3:7],
            ),
            axis=1,
        )
        self.assertLess(float(np.max(hard_attitude_error)), 1.0e-9)

        dense_parameters = np.linspace(0.0, 1.0, 5001)
        interpolating_curvature = _maximum_curvature(
            interpolating, dense_parameters
        )
        smoothed_curvature = _maximum_curvature(
            smoothed, dense_parameters
        )
        self.assertLess(
            smoothed_curvature, 0.5 * interpolating_curvature
        )
        self.assertGreater(smoothed.guide_position_rms_m, 0.0)

    def test_cubic_spline_interpolates_all_pose_nodes(self) -> None:
        positions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.4, 0.6, 1.2],
                [1.0, 0.2, 1.5],
                [1.4, -0.3, 1.1],
                [2.0, 0.0, 1.4],
            ]
        )
        quaternions = quaternion_from_euler(
            np.radians(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, -10.0, 35.0],
                    [-25.0, 30.0, 100.0],
                    [35.0, 15.0, 160.0],
                    [0.0, -20.0, -120.0],
                ]
            )
        )
        spline = InterpolatingSE3BSpline(
            np.concatenate((positions, quaternions), axis=1)
        )
        states = spline.evaluate(spline.parameters)
        np.testing.assert_allclose(
            states[:, :3], positions, atol=1.0e-10
        )
        attitude_error = np.linalg.norm(
            quaternion_error_vector(
                states[:, 3:7], quaternions
            ),
            axis=1,
        )
        self.assertLess(float(np.max(attitude_error)), 1.0e-9)
        np.testing.assert_allclose(
            np.linalg.norm(states[:, 3:7], axis=1),
            1.0,
            atol=1.0e-12,
        )

    def test_multi_segment_plan_and_timing(self) -> None:
        poses = (
            ((-1.5, -0.7, 1.0), (0.0, 0.0, -20.0)),
            ((-0.8, 0.8, 1.4), (25.0, -15.0, 30.0)),
            ((0.0, 1.2, 1.0), (-20.0, 25.0, 90.0)),
            ((0.8, -0.6, 1.6), (35.0, 10.0, 150.0)),
            ((1.5, 0.7, 1.3), (10.0, -20.0, -130.0)),
        )
        waypoints = tuple(
            SE3Pose(
                np.asarray(position),
                quaternion_from_euler(np.radians(rpy)),
            )
            for position, rpy in poses
        )
        planner = OMPLSE3Planner(
            bounds_min=(-2.0, -1.8, 0.2),
            bounds_max=(2.0, 1.8, 2.2),
            obstacles=(
                SphereObstacle(np.array([0.0, 0.0, 1.2]), 0.22),
            ),
            vehicle_radius=0.15,
            safety_margin=0.08,
            planner_range=0.35,
            seed=19,
        )
        plan = MultiWaypointOMPLPlanner(planner).plan(
            waypoints,
            solve_time_per_segment=0.5,
            minimum_states_per_segment=30,
            knot_stride=3,
            spline_samples=600,
            shortest_orientation_guide=True,
        )
        self.assertEqual(len(plan.segment_paths), 4)
        self.assertGreater(plan.minimum_clearance_m, 0.0)
        self.assertEqual(
            plan.spline_method, "waypoint-constrained smoothing"
        )
        self.assertLess(
            plan.control_point_count, len(plan.raw_states)
        )
        self.assertTrue(np.isfinite(plan.maximum_curvature_per_m))
        self.assertTrue(plan.orientation_shortcut_applied)

        waypoint_states = plan.spline.evaluate(
            plan.waypoint_parameters
        )
        np.testing.assert_allclose(
            waypoint_states[:, :3],
            np.asarray([pose.position for pose in waypoints]),
            atol=1.0e-7,
        )

        reference = BSplineTimeParameterizedReference(
            plan,
            max_linear_speed=0.9,
            max_angular_speed=1.3,
            duration_scale=1.08,
            timing_samples=2000,
        )
        times = np.linspace(
            0.0, reference.finish_time + 0.3, 6000
        )
        states = reference.sample(times)
        self.assertLessEqual(
            float(np.max(np.linalg.norm(states[:, 3:6], axis=1))),
            0.9 + 0.015,
        )
        self.assertLessEqual(
            float(
                np.max(np.linalg.norm(states[:, 10:13], axis=1))
            ),
            1.3 + 0.02,
        )
        np.testing.assert_allclose(
            states[[0, -1], 3:6], 0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            states[[0, -1], 10:13], 0.0, atol=1.0e-12
        )


def _maximum_curvature(
    spline: InterpolatingSE3BSpline, parameters: np.ndarray
) -> float:
    _, first, second, _, _ = (
        spline.evaluate_with_second_derivatives(parameters)
    )
    speed = np.linalg.norm(first, axis=1)
    curvature = np.divide(
        np.linalg.norm(np.cross(first, second), axis=1),
        speed**3,
        out=np.zeros_like(speed),
        where=speed > 1.0e-9,
    )
    return float(np.max(curvature))


if __name__ == "__main__":
    unittest.main()
