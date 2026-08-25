"""Quaternion compatibility namespace used by the standalone collector.

The collector reuses the HNUTER quaternion convention without importing the
MPPI controller, dynamics, Torch, or MuJoCo runtime.
"""

from .quaternion import (
    body_rates_from_euler_rates,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_error_vector,
    quaternion_from_euler,
    quaternion_from_rotation_vector,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)

__all__ = [
    "body_rates_from_euler_rates",
    "normalize_quaternion",
    "quaternion_conjugate",
    "quaternion_error_vector",
    "quaternion_from_euler",
    "quaternion_from_rotation_vector",
    "quaternion_multiply",
    "quaternion_to_euler",
    "quaternion_to_rotation_matrix",
]
