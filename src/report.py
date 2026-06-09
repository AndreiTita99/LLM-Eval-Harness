"""Reporting: build a structured report and render report.json + report.html.

One `build_report(...)` produces a plain, JSON-serialisable dict that is both the
machine artifact (`report.json`) and the data the Jinja2 HTML template consumes.
Keeping a single source of truth means the human and machine reports can never
drift apart.

The report carries everything a reviewer needs: the gate verdict, the
diff-vs-baseline (the bit you screenshot), per-scorer and overall pass-rate,
property metrics, flaky cases, and per-case detail — the model's actual output
for each repeat plus the judge's reasoning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .baseline import BaselineSnapshot, RegressionReport
from .config import Config
from .metrics import PropertyMetrics
from .models import EvalCase, RunSummary

_TEMPLATES = Path(__file__).parent / "templates"


def build_report(
    cases: list[EvalCase],
    summary: RunSummary,
    metrics: PropertyMetrics,
    config: Config,
    baseline: BaselineSnapshot | None,
    regression: RegressionReport | None,
) -> dict:
    """Assemble the full report as a JSON-serialisable dict."""
    scores_by_case_scorer: dict[tuple[str, str], list] = {}
    for s in summary.scores:
        scores_by_case_scorer.setdefault((s.case_id, s.scorer), []).append(s)

    results_by_case: dict[str, list] = {}
    for r in summary.results:
        results_by_case.setdefault(r.case_id, []).append(r)

    case_reports = []
    for case in cases:
        results = sorted(results_by_case.get(case.id, []), key=lambda r: r.repeat)
        scorer_rows = []
        # Preserve the order scorers were declared, then any extras (properties).
        seen: list[str] = []
        for name in [*case.scorers, "format_valid", "no_refusal"]:
            if name in seen or (case.id, name) not in scores_by_case_scorer:
                continue
            seen.append(name)
            lst = sorted(scores_by_case_scorer[(case.id, name)], key=lambda s: s.repeat)
            passed = sum(int(s.passed) for s in lst)
            scorer_rows.append({
                "scorer": name,
                "passed": passed,
                "total": len(lst),
                "pass_rate": passed / len(lst) if lst else 0.0,
                "is_flaky": len(lst) > 1 and 0 < passed < len(lst),
                "samples": [{"repeat": s.repeat, "passed": s.passed, "detail": s.detail} for s in lst],
            })
        case_reports.append({
            "id": case.id,
            "held_out": case.held_out,
            "input": case.input,
            "expected": case.expected,
            "all_passed": all(row["passed"] == row["total"] for row in scorer_rows) if scorer_rows else True,
            "scorers": scorer_rows,
            "repeats": [
                {
                    "repeat": r.repeat,
                    "parsed": r.parsed,
                    "raw_text": r.raw_text,
                    "latency_ms": round(r.latency_ms, 1),
                    "error": r.error,
                }
                for r in results
            ],
        })

    gate = {
        "has_baseline": baseline is not None,
        "passed": (regression.passed if regression is not None else None),
        "regressions": [d.name for d in regression.regressions] if regression is not None else [],
    }

    baseline_diff = None
    if regression is not None:
        baseline_diff = [
            {
                "name": d.name,
                "baseline": d.baseline,
                "current": d.current,
                "delta": d.delta,
                "higher_is_better": d.higher_is_better,
                "regressed": d.regressed,
                "note": d.note,
            }
            for d in regression.deltas
        ]

    overall = summary.overall()
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": config.provider,
            "sut_model": config.sut_model,
            "judge_model": config.judge_model,
            "repeats": config.repeats,
            "total_cases": summary.total_cases,
            "tolerances": {
                "accuracy_drop": config.accuracy_drop_tolerance,
                "latency_growth": config.latency_growth_tolerance,
                "cost_growth": config.cost_growth_tolerance,
            },
        },
        "gate": gate,
        "baseline": baseline.model_dump() if baseline is not None else None,
        "baseline_diff": baseline_diff,
        "overall": {"passed": overall.passed, "total": overall.total, "pass_rate": overall.pass_rate},
        "scorer_pass_rates": [
            {"scorer": s.scorer, "passed": s.passed, "total": s.total, "pass_rate": s.pass_rate}
            for s in summary.by_scorer()
        ],
        "flaky": [
            {"case_id": s.case_id, "scorer": s.scorer, "passed": s.passed, "total": s.total}
            for s in summary.flaky()
        ],
        "properties": {
            "n_calls": metrics.n_calls,
            "n_errors": metrics.n_errors,
            "latency_mean_ms": round(metrics.latency_mean_ms, 1),
            "latency_p95_ms": round(metrics.latency_p95_ms, 1),
            "total_input_tokens": metrics.total_input_tokens,
            "total_output_tokens": metrics.total_output_tokens,
            "estimated_cost_usd": round(metrics.estimated_cost_usd, 6),
        },
        "skipped_scorers": summary.skipped_scorers,
        "cases": case_reports,
    }


def render_html(report: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["pct"] = lambda v: f"{v * 100:.0f}%"
    return env.get_template("report.html.j2").render(report=report)


def write_reports(report: dict, report_dir: str | Path) -> tuple[Path, Path]:
    """Write report.json + report.html into report_dir; return their paths."""
    import json

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    html_path = out / "report.html"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path
