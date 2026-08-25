"""Command-line entry point for the task-pluggable expert collector."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .registry import create_default_registry
from .service import ExpertCollectorService
from .web import serve


def parse_args(
    argv: Sequence[str] | None = None, *, default_task: str = "free_flight",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--task", default=default_task)
    parser.add_argument(
        "--describe-tasks", action="store_true",
        help="print the task plugin catalog and exit",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="print one generated task condition and exit",
    )
    # Compatibility flags for the historical obstacle_scene_builder command.
    parser.add_argument("--family", default="staggered_corridor")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--obstacle-count", type=int, default=7)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, *, default_task: str = "free_flight",
) -> None:
    args = parse_args(argv, default_task=default_task)
    service = ExpertCollectorService(
        create_default_registry(), default_task=default_task,
    )
    if args.describe_tasks:
        print(json.dumps(service.list_tasks(), indent=2, ensure_ascii=False))
        return
    if args.generate:
        result = service.generate({
            "task_type": args.task,
            "family": args.family,
            "seed": args.seed,
            "obstacle_count": args.obstacle_count,
        })
        print(json.dumps(result["condition"], indent=2, ensure_ascii=False))
        return
    # Validate the selected task early, while still allowing planned task
    # descriptors to appear in discovery output and the UI.
    service.registry.get(args.task).ensure_available()
    serve(service, host=args.host, port=args.port)
