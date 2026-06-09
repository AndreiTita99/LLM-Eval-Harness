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

from .baseline import (
    BaselineSnapshot,
    RegressionReport,
    compare,
    load_baseline,
    save_baseline,
    snapshot,
)
from .config import Config
from .llm.judge import make_judge
from .metrics import PropertyMetrics, property_metrics
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


def _run_and_snapshot(
    config: Config, dataset: Path, prompt_path: Path
) -> tuple[RunSummary, PropertyMetrics, BaselineSnapshot]:
    cases = load_cases(dataset)
    prompt = load_prompt(prompt_path)
    summary = run(cases, prompt, config)
    metrics = property_metrics(summary.results, fallback_model=config.sut_model)
    return summary, metrics, snapshot(summary, metrics, config)


def _print_diff(report: RegressionReport) -> None:
    print("\nBaseline comparison (diff vs last known-good)")
    print("-" * 78)
    print(f"{'METRIC':<26} {'BASELINE':>10} {'CURRENT':>10} {'DELTA':>10}  STATUS")
    for d in report.deltas:
        status = "REGRESSED" if d.regressed else "ok"
        print(
            f"{d.name:<26} {d.baseline:>10.4g} {d.current:>10.4g} "
            f"{d.delta:>+10.4g}  {status}"
        )
    print("-" * 78)
    if report.passed:
        print("\nGATE PASS - no regression beyond tolerance.")
    else:
        names = ", ".join(d.name for d in report.regressions)
        print(f"\nGATE FAIL - {len(report.regressions)} regression(s): {names}")


def cmd_run(args: argparse.Namespace) -> int:
    config = Config.from_env()
    summary, _metrics, current = _run_and_snapshot(config, args.dataset, args.prompt)
    _print_summary(summary, config)

    if args.no_gate:
        return 0

    baseline_path = args.baseline or config.baseline_path
    baseline = load_baseline(baseline_path)
    if baseline is None:
        print(
            f"\nNo baseline at {baseline_path!r}; not gating. "
            "Establish one with `eval baseline update`."
        )
        return 0

    report = compare(current, baseline, config)
    _print_diff(report)
    return 0 if report.passed else 1


def cmd_baseline_update(args: argparse.Namespace) -> int:
    config = Config.from_env()
    summary, _metrics, current = _run_and_snapshot(config, args.dataset, args.prompt)
    _print_summary(summary, config)

    baseline_path = args.baseline or config.baseline_path
    save_baseline(current, baseline_path)
    print(
        f"\nBaseline updated -> {baseline_path}"
        f"\n  overall pass-rate: {current.overall_pass_rate:.2%}"
        f"   p95 latency: {current.latency_p95_ms:.0f} ms"
        f"   est. cost/call: ${current.cost_per_call_usd:.6f}"
    )
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

    run_p = sub.add_parser(
        "run", help="Run the golden set and gate against the baseline (exit 1 on regression)."
    )
    run_p.add_argument("--dataset", default=DEFAULT_DATASET, type=Path)
    run_p.add_argument("--prompt", default=DEFAULT_PROMPT, type=Path)
    run_p.add_argument("--baseline", default=None, help="Path to baseline.json (default from config).")
    run_p.add_argument("--no-gate", action="store_true", help="Run without comparing to the baseline.")
    run_p.set_defaults(func=cmd_run)

    baseline_p = sub.add_parser("baseline", help="Manage the regression baseline.")
    baseline_sub = baseline_p.add_subparsers(dest="baseline_cmd", required=True)
    update_p = baseline_sub.add_parser("update", help="Promote the current run to baseline.json.")
    update_p.add_argument("--dataset", default=DEFAULT_DATASET, type=Path)
    update_p.add_argument("--prompt", default=DEFAULT_PROMPT, type=Path)
    update_p.add_argument("--baseline", default=None, help="Path to write (default from config).")
    update_p.set_defaults(func=cmd_baseline_update)

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
