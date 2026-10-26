"""Battery-only V2 command-line entry point. Run `python main.py --help`."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from typing import Callable

from src.config import AppConfig, load_config, seed_everything
from src.cross_validation import cross_validate
from src.data_loader import inspect_dataset
from src.errors import ConfigError, DataError, StageNotImplementedError
from src.evaluation import ablation, compare_dual_path, compare_models, evaluate_low_load, research_evaluation
from src.features import generate_features
from src.fault_injection import generate_faults
from src.models import baseline
from src.preprocessing import preprocess
from src.replay import replay
from src.reporting import format_json
from src.tuning import tune, tune_dual_path


@dataclass(frozen=True)
class Command:
    handler: Callable[[AppConfig], dict]
    stage: int
    implemented: bool = False


COMMANDS = {
    "inspect": Command(inspect_dataset, 1, True),
    "preprocess": Command(preprocess, 2, True),
    "features": Command(generate_features, 3, True),
    "baseline": Command(baseline, 4, True),
    "inject-faults": Command(generate_faults, 5, True),
    "cross-validate": Command(cross_validate, 6, True),
    "ablation": Command(ablation, 7, True),
    "tune": Command(tune, 8, True),
    "compare-models": Command(compare_models, 7, True),
    "evaluate-low-load": Command(evaluate_low_load, 10, True),
    "tune-dual-path": Command(tune_dual_path, 9, True),
    "compare-dual-path": Command(compare_dual_path, 9, True),
    "research-evaluation": Command(research_evaluation, 11, True),
    "replay": Command(replay, 12, True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UAV battery-only anomaly detection V2 — inspect, preprocess, features, baseline, inject-faults, cross-validate")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML path relative to V2 root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, command in COMMANDS.items():
        status = "implemented" if command.implemented else f"not implemented; Stage {command.stage}"
        sub = subparsers.add_parser(name, help=status, description=f"{name}: {status}")
        # Accept --config both before and after the command without overwriting a global value.
        sub.add_argument("--config", default=argparse.SUPPRESS, help="YAML path relative to V2 root")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Use UTF-8 JSON/help consistently on Windows, including redirected output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        seed_everything(config.seed)
        payload = COMMANDS[args.command].handler(config)
    except StageNotImplementedError as exc:
        print(format_json({"status": "NOT_IMPLEMENTED", "command": exc.command,
                           "required_stage": exc.stage, "error": str(exc)}), file=sys.stderr)
        return 3
    except (ConfigError, DataError, OSError) as exc:
        print(format_json({"status": "ERROR", "command": args.command, "error": str(exc)}), file=sys.stderr)
        return 2
    print(format_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
