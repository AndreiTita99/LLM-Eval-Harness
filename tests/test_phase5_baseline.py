"""Phase 5 tests — baseline snapshot, persistence, regression gating, exit codes."""

from __future__ import annotations

from src.baseline import (
    BaselineSnapshot,
    compare,
    load_baseline,
    save_baseline,
    snapshot,
)
from src.cli import main
from src.config import Config
from src.metrics import property_metrics
from src.models import EvalCase
from src.runner import run


def _snap(scorer_rates, overall, latency=100.0, cost=0.001) -> BaselineSnapshot:
    return BaselineSnapshot(
        created_at="t", provider="mock", sut_model="m", repeats=1, total_cases=1,
        scorer_pass_rates=scorer_rates, overall_pass_rate=overall,
        latency_p95_ms=latency, cost_per_call_usd=cost,
    )


def _cfg() -> Config:
    return Config(provider="mock")  # acc tol 0.02, latency/cost tol 0.20


# --- snapshot + persistence -------------------------------------------------

def test_snapshot_from_run(tmp_path):
    config = Config(provider="mock", repeats=1)
    cases = [EvalCase(id="t1", input="My card was charged twice", expected={"category": "billing"},
                      scorers=["category_exact"])]
    summary = run(cases, "p", config)
    metrics = property_metrics(summary.results, fallback_model=config.sut_model)
    snap = snapshot(summary, metrics, config)
    assert snap.scorer_pass_rates["category_exact"] == 1.0
    assert snap.overall_pass_rate == 1.0
    assert snap.cost_per_call_usd > 0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    snap = _snap({"category_exact": 1.0}, 1.0)
    save_baseline(snap, path)
    loaded = load_baseline(path)
    assert loaded == snap


def test_load_missing_returns_none(tmp_path):
    assert load_baseline(tmp_path / "nope.json") is None


# --- comparison logic -------------------------------------------------------

def test_identical_snapshot_passes():
    snap = _snap({"category_exact": 1.0, "summary_judge": 0.9}, 0.95)
    report = compare(snap, snap, _cfg())
    assert report.passed is True
    assert report.regressions == []


def test_pass_rate_drop_beyond_tolerance_regresses():
    base = _snap({"category_exact": 1.0}, 1.0)
    cur = _snap({"category_exact": 0.90}, 0.90)  # -0.10, tol is 0.02
    report = compare(cur, base, _cfg())
    assert report.passed is False
    assert any(d.name == "pass_rate[category_exact]" and d.regressed for d in report.deltas)


def test_pass_rate_drop_within_tolerance_ok():
    base = _snap({"category_exact": 1.0}, 1.0)
    cur = _snap({"category_exact": 0.99}, 0.99)  # -0.01, within 0.02
    assert compare(cur, base, _cfg()).passed is True


def test_missing_scorer_in_current_regresses():
    base = _snap({"category_exact": 1.0}, 1.0)
    cur = _snap({}, 1.0)  # scorer disappeared -> treated as 0.0
    report = compare(cur, base, _cfg())
    assert any(d.name == "pass_rate[category_exact]" and d.regressed for d in report.deltas)


def test_latency_growth_gate():
    base = _snap({"s": 1.0}, 1.0, latency=100.0)
    assert compare(_snap({"s": 1.0}, 1.0, latency=130.0), base, _cfg()).passed is False  # +30% > 20%
    assert compare(_snap({"s": 1.0}, 1.0, latency=115.0), base, _cfg()).passed is True   # +15% < 20%


def test_cost_growth_gate():
    base = _snap({"s": 1.0}, 1.0, cost=0.0010)
    assert compare(_snap({"s": 1.0}, 1.0, cost=0.0013), base, _cfg()).passed is False  # +30%
    assert compare(_snap({"s": 1.0}, 1.0, cost=0.0011), base, _cfg()).passed is True   # +10%


def test_zero_baseline_latency_never_regresses():
    base = _snap({"s": 1.0}, 1.0, latency=0.0)
    assert compare(_snap({"s": 1.0}, 1.0, latency=5.0), base, _cfg()).passed is True


# --- CLI exit codes (end to end) -------------------------------------------

def test_cli_baseline_update_then_gate(tmp_path, monkeypatch, capsys):
    bp = tmp_path / "baseline.json"
    assert main(["baseline", "update", "--baseline", str(bp)]) == 0
    assert bp.exists()
    # Clean run against the just-written baseline -> gate passes.
    assert main(["run", "--baseline", str(bp)]) == 0
    # A flaky run degrades pass-rate -> gate fails with a non-zero exit code.
    monkeypatch.setenv("EVAL_MOCK_FLAKINESS", "0.6")
    monkeypatch.setenv("EVAL_REPEATS", "6")
    assert main(["run", "--baseline", str(bp)]) == 1
    capsys.readouterr()  # swallow output


def test_cli_run_no_gate_returns_zero(tmp_path):
    # --no-gate skips comparison entirely, even with no baseline.
    assert main(["run", "--no-gate", "--baseline", str(tmp_path / "absent.json")]) == 0
