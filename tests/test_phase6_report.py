"""Phase 6 tests — report building, HTML rendering, and file output."""

from __future__ import annotations

import json

from src.baseline import compare, snapshot
from src.config import Config
from src.metrics import property_metrics
from src.models import EvalCase
from src.report import build_report, render_html, write_reports
from src.runner import run


def _run(config=None):
    config = config or Config(provider="mock", repeats=2)
    cases = [
        EvalCase(id="t1", input="My card was charged twice for order 4471",
                 expected={"category": "billing", "urgency": "high"},
                 scorers=["category_exact", "urgency_schema", "summary_judge"]),
        EvalCase(id="t2", input="How do I change my account email?", expected={"category": "account"},
                 scorers=["category_exact"], held_out=True),
    ]
    summary = run(cases, "p", config)
    metrics = property_metrics(summary.results, fallback_model=config.sut_model)
    return cases, summary, metrics, config


def test_build_report_without_baseline():
    cases, summary, metrics, config = _run()
    report = build_report(cases, summary, metrics, config, baseline=None, regression=None)

    assert report["gate"]["has_baseline"] is False
    assert report["baseline_diff"] is None
    assert len(report["cases"]) == 2
    assert report["properties"]["n_calls"] == 4  # 2 cases x 2 repeats
    # held-out flag and property scorers both surface in the per-case detail.
    t2 = next(c for c in report["cases"] if c["id"] == "t2")
    assert t2["held_out"] is True
    assert {s["scorer"] for s in t2["scorers"]} >= {"category_exact", "format_valid", "no_refusal"}


def test_build_report_with_baseline_and_gate():
    cases, summary, metrics, config = _run()
    current = snapshot(summary, metrics, config)
    regression = compare(current, current, config)  # compare to self -> passes
    report = build_report(cases, summary, metrics, config, baseline=current, regression=regression)

    assert report["gate"]["has_baseline"] is True
    assert report["gate"]["passed"] is True
    assert report["baseline_diff"] is not None
    assert any(d["name"] == "overall_pass_rate" for d in report["baseline_diff"])


def test_judge_reasoning_present_in_case_detail():
    cases, summary, metrics, config = _run()
    report = build_report(cases, summary, metrics, config, None, None)
    t1 = next(c for c in report["cases"] if c["id"] == "t1")
    judge_row = next(s for s in t1["scorers"] if s["scorer"] == "summary_judge")
    assert "score=" in judge_row["samples"][0]["detail"]


def test_render_html_contains_key_sections():
    cases, summary, metrics, config = _run()
    current = snapshot(summary, metrics, config)
    report = build_report(cases, summary, metrics, config, current, compare(current, current, config))
    html = render_html(report)
    assert "LLM Evaluation Report" in html
    assert "Diff vs baseline" in html
    assert "GATE PASS" in html
    assert "t1" in html and "t2" in html


def test_write_reports_creates_both_files(tmp_path):
    cases, summary, metrics, config = _run()
    report = build_report(cases, summary, metrics, config, None, None)
    json_path, html_path = write_reports(report, tmp_path / "out")

    assert json_path.exists() and html_path.exists()
    # report.json round-trips to the same structure.
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["meta"]["total_cases"] == 2
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
