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
from .llm.judge import make_judge
from .metrics import property_metrics
from .models import RunSummary
from .runner import load_cases, load_prompt, run
from .validation import ValidationReport, load_labeled, validate_judge

DEFAULT_DATASET = "datasets/triage.yaml"
DEFAULT_PROMPT = "prompts/triage_v1.txt"
DEFAULT_LABELED = "datasets/judge_labeled.yaml"


def _print_summary(summary: RunSummary, config: Config) -> None:
    print(f"\nProvider: {config.provider}    SUT model: {config.sut_model}")
    print(f"Cases: {summary.total_cases}    Repeats per case: {config.repeats}\n")

    # Per (case, scorer) pass-rate across repeats — variance is folded in here.
    print(f"{'CASE':<12} {'SCORER':<16} {'PASS-RATE':<10} FLAG")
    print("-" * 78)
    for stat in summary.by_case_scorer():
        rate = f"{stat.passed}/{stat.total}"
        flag = "FLAKY" if stat.is_flaky else ""
        print(f"{stat.case_id:<12} {stat.scorer:<16} {rate:<10} {flag}")

    print("\nPer-scorer pass-rate (across all cases x repeats)")
    print("-" * 78)
    for stat in summary.by_scorer():
        print(f"  {stat.scorer:<18} {stat.passed:>3}/{stat.total:<4} ({stat.pass_rate * 100:.0f}%)")

    flaky = summary.flaky()
    if flaky:
        print(f"\nFlaky (non-deterministic) cases: {len(flaky)}")
        for s in flaky:
            print(f"  {s.case_id} / {s.scorer}: {s.passed}/{s.total}")
    else:
        print("\nFlaky cases: none")

    metrics = property_metrics(summary.results, fallback_model=config.sut_model)
    print("\nProperties")
    print("-" * 78)
    print(f"  SUT calls:        {metrics.n_calls}  (errors: {metrics.n_errors})")
    print(f"  Latency mean/p95: {metrics.latency_mean_ms:.0f} ms / {metrics.latency_p95_ms:.0f} ms")
    print(f"  Tokens in/out:    {metrics.total_input_tokens} / {metrics.total_output_tokens}")
    print(f"  Est. cost:        ${metrics.estimated_cost_usd:.4f}")

    if summary.skipped_scorers:
        print(f"\nSkipped (no scorer registered): {', '.join(summary.skipped_scorers)}")

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


def _print_validation(report: ValidationReport, config: Config) -> None:
    print(f"\nJudge validation    provider: {config.provider}    judge model: {config.judge_model}")
    print(f"Pass threshold: score >= {report.threshold}    examples: {report.n}\n")

    print(f"{'ID':<6} {'HUMAN':<6} {'JUDGE':<6} {'EXACT':<6} {'PASS/FAIL':<10} REASONING")
    print("-" * 86)
    for c in report.comparisons:
        judge = "?" if c.judge_score is None else str(c.judge_score)
        print(
            f"{c.id:<6} {c.human_score:<6} {judge:<6} "
            f"{('=' if c.agree_exact else 'x'):<6} "
            f"{('agree' if c.agree_passfail else 'DIFFER'):<10} {c.reasoning[:40]}"
        )

    print("-" * 86)
    print(f"\nExact agreement:     {report.exact_agreement * 100:.0f}%")
    print(f"Pass/fail agreement: {report.passfail_agreement * 100:.0f}%")
    print(f"Cohen's kappa:       {report.kappa:.2f}  ({_kappa_label(report.kappa)})")


def _kappa_label(k: float) -> str:
    if k < 0:
        return "worse than chance"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


def cmd_judge_validate(args: argparse.Namespace) -> int:
    config = Config.from_env()
    labeled = load_labeled(args.dataset)
    judge = make_judge(config)
    report = validate_judge(judge, labeled, threshold=config.judge_pass_threshold)
    _print_validation(report, config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description="CI for prompts.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the golden set through the SUT prompt.")
    run_p.add_argument("--dataset", default=DEFAULT_DATASET, type=Path)
    run_p.add_argument("--prompt", default=DEFAULT_PROMPT, type=Path)
    run_p.set_defaults(func=cmd_run)

    jv_p = sub.add_parser(
        "judge-validate",
        help="Measure the judge's agreement with human labels on a labelled set.",
    )
    jv_p.add_argument("--dataset", default=DEFAULT_LABELED, type=Path)
    jv_p.set_defaults(func=cmd_judge_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
