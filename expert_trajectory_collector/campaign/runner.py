"""Environment-level multiprocessing orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .config import BatchConfig
from .free_flight import collect_environment
from .io import atomic_write_json, read_json
from .monitor import start_monitor
from .state import aggregate_status


def _software_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_revision": revision,
        "visualization_enabled": False,
        "toppra_enabled": False,
        "mujoco_rollout_enabled": False,
    }


def prepare_campaign(root: Path, config: BatchConfig, resume: bool) -> None:
    config.validate()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign_config.json"
    existing = read_json(path)
    if existing is not None:
        if not resume:
            raise FileExistsError(
                f"campaign already exists at {root}; pass --resume to continue"
            )
        previous = BatchConfig.from_dict(existing)
        if previous.dataset_signature() != config.dataset_signature():
            raise ValueError("resume configuration changes dataset-defining fields")
    else:
        atomic_write_json(path, config.to_dict())
        atomic_write_json(root / "provenance.json", _software_provenance())
    runtime = read_json(root / "runtime.json", {}) or {}
    runtime.setdefault("started_at_unix_s", time.time())
    runtime["last_started_at_unix_s"] = time.time()
    runtime["workers"] = config.workers
    atomic_write_json(root / "runtime.json", runtime)


def run_campaign(
    root: Path, config: BatchConfig, *, resume: bool = False,
    monitor_host: str = "127.0.0.1", monitor_port: int = 8785,
    enable_monitor: bool = True,
) -> dict[str, Any]:
    prepare_campaign(root, config, resume)
    running_marker = root / "RUNNING"
    if running_marker.exists():
        try:
            existing_pid = int(running_marker.read_text(encoding="utf-8").strip())
            os.kill(existing_pid, 0)
        except (OSError, ValueError):
            running_marker.unlink(missing_ok=True)
        else:
            if existing_pid != os.getpid():
                raise RuntimeError(
                    f"campaign already has a live collector process (PID {existing_pid})"
                )
    running_marker.write_text(str(os.getpid()), encoding="utf-8")
    server = None
    if enable_monitor:
        server, _ = start_monitor(root, monitor_host, monitor_port)
        print(f"monitor: http://{monitor_host}:{monitor_port}", flush=True)
    completed = {
        int(path.parent.name.split("_")[-1])
        for path in (root / "environments").glob("env_*/progress.json")
        if (read_json(path, {}) or {}).get("worker_state") == "complete"
    } if (root / "environments").exists() else set()
    remaining = [index for index in range(config.environment_count) if index not in completed]
    last_report = 0.0
    try:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(collect_environment, config.to_dict(), str(root), index): index
                for index in remaining
            }
            while futures:
                done_now = []
                for future in list(futures):
                    if future.done():
                        future.result()
                        done_now.append(future)
                for future in done_now:
                    futures.pop(future, None)
                now = time.time()
                if now - last_report >= 2.0 or not futures:
                    status = aggregate_status(root)
                    eta = status["eta_s"]
                    eta_text = "--" if eta is None else f"{eta / 60:.1f} min"
                    print(
                        f"[{status['state']}] env {status['environment_complete']}/{status['environment_target']} "
                        f"paths {status['path_accepted']}/{status['path_target']} "
                        f"rate {status['accepted_paths_per_s']:.3f}/s ETA {eta_text}",
                        flush=True,
                    )
                    last_report = now
                if futures:
                    time.sleep(0.25)
    finally:
        running_marker.unlink(missing_ok=True)
        if server is not None:
            server.shutdown()
            server.server_close()
    status = aggregate_status(root)
    manifest = {
        "schema_version": "expert_collection_campaign_manifest_v001",
        "config": config.to_dict(),
        "status": status,
        "provenance": read_json(root / "provenance.json", {}),
        "storage": {
            "environment": "environments/env_N/environment.json",
            "condition": "environments/env_N/conditions/condition_N/condition.json.gz",
            "metadata": "environments/env_N/conditions/condition_N/expert_metadata.json",
            "arrays": "environments/env_N/conditions/condition_N/paths.npz",
        },
    }
    atomic_write_json(root / "dataset_manifest.json", manifest)
    return status
