"""Command-line entry point.

Phase 1 exposes a single command:

    eval run    # run the golden set through the SUT and print pass/fail

(Run as `python -m src.cli run` without installing.) Later phases add
`eval baseline update` and a non-zero exit code on regression.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .models import RunSummary
from .runner import load_cases, load_prompt, run

DEFAULT_DATASET = "datasets/triage.yaml"
DEFAULT_PROMPT = "prompts/triage_v1.txt"


def _print_summary(summary: RunSummary, config: Config) -> None:
    print(f"\nProvider: {config.provider}    SUT model: {config.sut_model}")
    print(f"Cases: {summary.total_cases}\n")

    print(f"{'CASE':<12} {'SCORER':<16} {'RESULT':<7} DETAIL")
    print("-" * 78)
    for score in summary.scores:
        mark = "PASS" if score.passed else "FAIL"
        print(f"{score.case_id:<12} {score.scorer:<16} {mark:<7} {score.detail}")

    print("\nPer-scorer pass-rate")
    print("-" * 78)
    for stat in summary.by_scorer():
        print(f"  {stat.scorer:<18} {stat.passed:>3}/{stat.total:<3} ({stat.pass_rate * 100:.0f}%)")

    if summary.skipped_scorers:
        print(f"\nSkipped (not registered yet): {', '.join(summary.skipped_scorers)}")

    overall = summary.overall()
    print("-" * 78)
    print(f"\nOverall: {overall.passed}/{overall.total} checks passed ({overall.pass_rate * 100:.0f}%)")


def cmd_run(args: argparse.Namespace) -> int:
    config = Config.from_env()
    cases = load_cases(args.dataset)
    prompt = load_prompt(args.prompt)
    summary = run(cases, prompt, config)
    _print_summary(summary, config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description="CI for prompts.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the golden set through the SUT prompt.")
    run_p.add_argument("--dataset", default=DEFAULT_DATASET, type=Path)
    run_p.add_argument("--prompt", default=DEFAULT_PROMPT, type=Path)
    run_p.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
