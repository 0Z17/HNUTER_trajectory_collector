"""Atomic campaign state and compact trajectory serialization."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .encoding import pose7_to_pose9, resample_pose7_path


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
    os.replace(temporary, path)


def save_expert_set(
    directory: Path, condition: dict[str, Any], expert_set: dict[str, Any],
    training_path_points: int, collision_checker: Any | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_gzip_json(directory / "condition.json.gz", condition)
    experts = expert_set.get("experts", [])
    ompl_values: list[np.ndarray] = []
    offsets = [0]
    smooth_values: list[np.ndarray] = []
    training_values: list[np.ndarray] = []
    clearance_values: list[np.ndarray] = []
    progress = np.linspace(0.0, 1.0, training_path_points, dtype=np.float32)
    metadata = []
    for expert in experts:
        raw = np.asarray(expert["ompl_path"], dtype=np.float32)
        smooth = np.asarray(expert["bspline_path"], dtype=np.float32)
        sampled, progress = resample_pose7_path(smooth, training_path_points)
        ompl_values.append(raw)
        offsets.append(offsets[-1] + len(raw))
        smooth_values.append(smooth)
        training_values.append(pose7_to_pose9(sampled))
        if collision_checker is not None:
            clearance_values.append(np.asarray(
                collision_checker.clearance(sampled[:, :3], sampled[:, 3:7]),
                dtype=np.float32,
            ))
        metadata.append({
            key: value for key, value in expert.items()
            if key not in {"ompl_path", "bspline_path"}
        })
    np.savez(
        directory / "paths.npz",
        ompl_path_pose7=np.concatenate(ompl_values, axis=0) if ompl_values else np.empty((0, 7), np.float32),
        ompl_path_offsets=np.asarray(offsets, dtype=np.int64),
        geometry_path_pose7=np.stack(smooth_values) if smooth_values else np.empty((0, 0, 7), np.float32),
        training_path_pose9=np.stack(training_values) if training_values else np.empty((0, training_path_points, 9), np.float32),
        normalized_arc_progress=progress,
        clearance_m=np.stack(clearance_values) if clearance_values else np.empty((0, training_path_points), np.float32),
    )
    compact = {
        key: value for key, value in expert_set.items() if key != "experts"
    }
    compact["experts"] = metadata
    compact["array_schema"] = {
        "file": "paths.npz",
        "ompl_path_pose7": "flattened [sum(Nraw),7], offsets indexed",
        "geometry_path_pose7": "[K,256,7] wxyz",
        "training_path_pose9": f"[K,{training_path_points},9] xyz+rotation6d",
        "clearance_m": f"[K,{training_path_points}] full URDF/COAL clearance",
    }
    atomic_write_json(directory / "expert_metadata.json", compact)
