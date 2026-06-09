"""Baseline tracking and regression gating.

This is what makes the harness a *CI gate* rather than a dashboard. A run is
snapshotted into a small set of aggregate metrics; `baseline.json` holds the
last known-good snapshot. A new run is compared to it within tolerances, and any
metric that moves the wrong way beyond its tolerance is a **regression** — which
the CLI turns into a non-zero exit code so a PR can't merge.

Core principle: gate on regression, not perfection. We don't require 100%
accuracy; we require that a change doesn't make quality, latency, or cost
*worse* than the last reviewed-good state beyond an allowed margin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from .config import Config
from .metrics import PropertyMetrics
from .models import RunSummary


class BaselineSnapshot(BaseModel):
    """The aggregate metrics we gate on, plus provenance metadata."""

    created_at: str
    provider: str
    sut_model: str
    repeats: int
    total_cases: int
    scorer_pass_rates: dict[str, float]
    overall_pass_rate: float
    latency_p95_ms: float
    # Cost is stored *per SUT call*, not per run, so the gate is invariant to how
    # many repeats a given run used (total cost scales with repeats; per-call
    # cost is what actually regresses when a prompt/model gets more expensive).
    cost_per_call_usd: float


def snapshot(summary: RunSummary, metrics: PropertyMetrics, config: Config) -> BaselineSnapshot:
    """Reduce a full run down to the gated aggregate metrics."""
    cost_per_call = metrics.estimated_cost_usd / metrics.n_calls if metrics.n_calls else 0.0
    return BaselineSnapshot(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        provider=config.provider,
        sut_model=config.sut_model,
        repeats=config.repeats,
        total_cases=summary.total_cases,
        scorer_pass_rates={s.scorer: round(s.pass_rate, 4) for s in summary.by_scorer()},
        overall_pass_rate=round(summary.overall().pass_rate, 4),
        latency_p95_ms=round(metrics.latency_p95_ms, 3),
        cost_per_call_usd=round(cost_per_call, 8),
    )


def save_baseline(snap: BaselineSnapshot, path: str | Path) -> None:
    Path(path).write_text(snap.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> BaselineSnapshot | None:
    """Load the baseline snapshot, or None if no baseline has been set yet."""
    p = Path(path)
    if not p.exists():
        return None
    return BaselineSnapshot.model_validate_json(p.read_text(encoding="utf-8"))


class MetricDelta(BaseModel):
    """One metric's movement from baseline to current, and whether it regressed."""

    name: str
    baseline: float
    current: float
    higher_is_better: bool
    regressed: bool
    note: str = ""

    @property
    def delta(self) -> float:
        return self.current - self.baseline


class RegressionReport(BaseModel):
    deltas: list[MetricDelta]

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.regressed]

    @property
    def passed(self) -> bool:
        return not self.regressions


def compare(current: BaselineSnapshot, baseline: BaselineSnapshot, config: Config) -> RegressionReport:
    """Compare a current snapshot to the baseline within configured tolerances."""
    deltas: list[MetricDelta] = []
    acc_tol = config.accuracy_drop_tolerance

    # Per-scorer pass-rates: higher is better; regress if it drops beyond tolerance.
    # A scorer present in the baseline but missing now reads as 0.0 — a regression,
    # which is the right call (a metric silently disappeared).
    for scorer, base_rate in sorted(baseline.scorer_pass_rates.items()):
        cur = current.scorer_pass_rates.get(scorer, 0.0)
        deltas.append(
            MetricDelta(
                name=f"pass_rate[{scorer}]",
                baseline=base_rate,
                current=cur,
                higher_is_better=True,
                regressed=cur < base_rate - acc_tol,
                note=f"max drop {acc_tol:.0%}-pts",
            )
        )

    deltas.append(
        MetricDelta(
            name="overall_pass_rate",
            baseline=baseline.overall_pass_rate,
            current=current.overall_pass_rate,
            higher_is_better=True,
            regressed=current.overall_pass_rate < baseline.overall_pass_rate - acc_tol,
            note=f"max drop {acc_tol:.0%}-pts",
        )
    )

    # Latency p95 and cost: lower is better; regress if they grow beyond tolerance.
    deltas.append(_growth_delta(
        "latency_p95_ms", baseline.latency_p95_ms, current.latency_p95_ms, config.latency_growth_tolerance,
    ))
    deltas.append(_growth_delta(
        "cost_per_call_usd", baseline.cost_per_call_usd, current.cost_per_call_usd, config.cost_growth_tolerance,
    ))
    return RegressionReport(deltas=deltas)


def _growth_delta(name: str, base: float, cur: float, tol: float) -> MetricDelta:
    # No regression possible against a zero/absent baseline value.
    regressed = base > 0 and cur > base * (1 + tol)
    return MetricDelta(
        name=name,
        baseline=base,
        current=cur,
        higher_is_better=False,
        regressed=regressed,
        note=f"max growth {tol:.0%}",
    )
