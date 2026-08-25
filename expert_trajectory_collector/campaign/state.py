"""Campaign control markers and aggregate read-only status."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from .io import atomic_write_json, read_json


TERMINAL_STATES = {"complete", "capacity_exhausted", "failed"}


def set_paused(root: Path, paused: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "PAUSED"
    if paused:
        marker.touch(exist_ok=True)
    else:
        marker.unlink(missing_ok=True)


def aggregate_status(root: Path) -> dict[str, Any]:
    config = read_json(root / "campaign_config.json", {}) or {}
    environment_target = int(config.get("environment_count", 0))
    path_target = environment_target * int(config.get("paths_per_environment", 0))
    items = []
    environment_root = root / "environments"
    if environment_root.exists():
        for path in sorted(environment_root.glob("env_*/progress.json")):
            value = read_json(path, {}) or {}
            if value:
                items.append(value)
    counts: dict[str, int] = {}
    for item in items:
        state = str(item.get("worker_state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    accepted = sum(int(item.get("accepted_path_count", 0)) for item in items)
    attempts = sum(int(item.get("planner_attempt_count", 0)) for item in items)
    started = float(read_json(root / "runtime.json", {}).get("started_at_unix_s", time.time()))
    elapsed = max(0.0, time.time() - started)
    rate = accepted / elapsed if elapsed > 0 else 0.0
    remaining = max(0, path_target - accepted)
    status = {
        "schema_version": "expert_collection_campaign_status_v001",
        "dataset_id": config.get("dataset_id"),
        "state": (
            "paused" if (root / "PAUSED").exists()
            else "complete" if environment_target and counts.get("complete", 0) == environment_target
            else "running" if (root / "RUNNING").exists()
            else "stopped"
        ),
        "paused": (root / "PAUSED").exists(),
        "environment_target": environment_target,
        "environment_started": len(items),
        "environment_complete": counts.get("complete", 0),
        "environment_capacity_exhausted": counts.get("capacity_exhausted", 0),
        "environment_failed": counts.get("failed", 0),
        "environment_states": counts,
        "path_target": path_target,
        "path_accepted": accepted,
        "planner_attempt_count": attempts,
        "elapsed_s": elapsed,
        "accepted_paths_per_s": rate,
        "eta_s": remaining / rate if rate > 0 else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(root / "status.json", status)
    return status
