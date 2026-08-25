"""CLI for recoverable headless collection campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .campaign.config import BatchConfig
from .campaign.io import read_json
from .campaign.runner import run_campaign
from .campaign.state import aggregate_status, set_paused


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="create or resume a campaign")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dataset-id", default="free_flight_pilot_v002")
    run.add_argument("--seed", type=int, default=20260825)
    run.add_argument("--environment-count", type=int, default=100)
    run.add_argument("--paths-per-environment", type=int, default=40)
    run.add_argument("--nominal-conditions-per-environment", type=int, default=8)
    run.add_argument("--experts-per-condition", type=int, default=5)
    run.add_argument("--maximum-conditions-per-environment", type=int, default=48)
    run.add_argument("--condition-sampling-max-attempts", type=int, default=64)
    run.add_argument("--condition-sampling-timeout", type=float, default=4.0)
    run.add_argument("--environment-precheck-condition-count", type=int, default=4)
    run.add_argument("--environment-precheck-minimum-successes", type=int, default=2)
    run.add_argument("--environment-precheck-max-attempts", type=int, default=24)
    run.add_argument("--environment-precheck-timeout", type=float, default=1.0)
    run.add_argument("--maximum-consecutive-condition-failures", type=int, default=8)
    run.add_argument("--terminal-attitude-margin-deg", type=float, default=5.0)
    run.add_argument("--workers", type=int, default=12)
    run.add_argument("--solve-time", type=float, default=0.20)
    run.add_argument("--maximum-planner-attempts", type=int, default=12)
    run.add_argument("--obstacle-count-min", type=int, default=6)
    run.add_argument("--obstacle-count-max", type=int, default=18)
    run.add_argument("--families", nargs="+")
    run.add_argument("--monitor-host", default="127.0.0.1")
    run.add_argument("--monitor-port", type=int, default=8785)
    run.add_argument("--no-monitor", action="store_true")
    for command in ("status", "pause", "resume"):
        item = subparsers.add_parser(command)
        item.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "status":
        print(json.dumps(aggregate_status(args.output), ensure_ascii=False, indent=2))
        return
    if args.command in {"pause", "resume"}:
        set_paused(args.output, args.command == "pause")
        print(json.dumps(aggregate_status(args.output), ensure_ascii=False, indent=2))
        return
    defaults = BatchConfig()
    config = BatchConfig(
        dataset_id=args.dataset_id,
        base_seed=args.seed,
        environment_count=args.environment_count,
        paths_per_environment=args.paths_per_environment,
        nominal_conditions_per_environment=args.nominal_conditions_per_environment,
        experts_per_condition=args.experts_per_condition,
        maximum_conditions_per_environment=args.maximum_conditions_per_environment,
        condition_sampling_max_attempts=args.condition_sampling_max_attempts,
        condition_sampling_timeout_s=args.condition_sampling_timeout,
        environment_precheck_condition_count=args.environment_precheck_condition_count,
        environment_precheck_minimum_successes=args.environment_precheck_minimum_successes,
        environment_precheck_max_attempts=args.environment_precheck_max_attempts,
        environment_precheck_timeout_s=args.environment_precheck_timeout,
        maximum_consecutive_condition_failures=args.maximum_consecutive_condition_failures,
        terminal_attitude_margin_deg=args.terminal_attitude_margin_deg,
        workers=args.workers,
        solve_time_s=args.solve_time,
        maximum_planner_attempts=args.maximum_planner_attempts,
        obstacle_count_min=args.obstacle_count_min,
        obstacle_count_max=args.obstacle_count_max,
        families=args.families or defaults.families,
    )
    if args.resume:
        stored = read_json(args.output / "campaign_config.json")
        if stored is None:
            raise FileNotFoundError("--resume requested but campaign_config.json is missing")
        stored["workers"] = args.workers
        config = BatchConfig.from_dict(stored)
    status = run_campaign(
        args.output, config, resume=args.resume,
        monitor_host=args.monitor_host, monitor_port=args.monitor_port,
        enable_monitor=not args.no_monitor,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
