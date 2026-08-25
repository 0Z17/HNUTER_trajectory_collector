"""Canonical path resampling and pose7-to-pose9 conversion."""

from __future__ import annotations

import numpy as np


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    for index in range(1, len(result)):
        if float(result[index - 1] @ result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.clip(first @ second, -1.0, 1.0))
    if dot < 0.0:
        second, dot = -second, -dot
    if dot > 0.9995:
        value = (1.0 - alpha) * first + alpha * second
        return value / max(float(np.linalg.norm(value)), 1e-12)
    angle = float(np.arccos(dot))
    scale = np.sin(angle)
    return (
        np.sin((1.0 - alpha) * angle) / scale * first
        + np.sin(alpha * angle) / scale * second
    )


def resample_pose7_path(
    path: np.ndarray, count: int = 128, rotation_weight_m_per_rad: float = 0.22,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(path, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7 or len(states) < 2:
        raise ValueError("pose7 path must have shape [N>=2, 7]")
    quaternions = _normalize_quaternions(states[:, 3:7])
    positions = states[:, :3]
    segment_position = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dots = np.clip(np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1)), 0.0, 1.0)
    segment_rotation = 2.0 * np.arccos(dots)
    segment_length = np.sqrt(
        segment_position**2 + (rotation_weight_m_per_rad * segment_rotation) ** 2
    )
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    if cumulative[-1] <= 1e-12:
        raise ValueError("cannot resample a zero-length path")
    targets = np.linspace(0.0, float(cumulative[-1]), int(count))
    output = np.empty((int(count), 7), dtype=np.float64)
    for output_index, target in enumerate(targets):
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(states) - 2)
        segment = max(segment, 0)
        denominator = cumulative[segment + 1] - cumulative[segment]
        alpha = 0.0 if denominator <= 1e-12 else float((target - cumulative[segment]) / denominator)
        output[output_index, :3] = (
            (1.0 - alpha) * positions[segment] + alpha * positions[segment + 1]
        )
        output[output_index, 3:7] = _slerp(
            quaternions[segment], quaternions[segment + 1], alpha,
        )
    progress = (targets / cumulative[-1]).astype(np.float32)
    return output.astype(np.float32), progress


def quaternion_wxyz_to_rotation6d(quaternions: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions(np.asarray(quaternions, dtype=np.float64))
    w, x, y, z = (q[:, index] for index in range(4))
    rotation = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)
    return np.concatenate((rotation[:, :, 0], rotation[:, :, 1]), axis=1).astype(np.float32)


def pose7_to_pose9(path: np.ndarray) -> np.ndarray:
    states = np.asarray(path, dtype=np.float32)
    return np.concatenate(
        (states[:, :3], quaternion_wxyz_to_rotation6d(states[:, 3:7])), axis=1,
    ).astype(np.float32)
