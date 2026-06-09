"""Property metrics — latency, token usage, and estimated cost per run.

These are continuous metrics (not pass/fail), aggregated across all SUT calls in
a run. They're first-class outputs: latency p95 and cost are exactly the kind of
thing a prompt or model change can regress without touching accuracy, and Phase 5
gates on them.

Cost is an *estimate* from a static price table. The mock provider reports a
`model` of "mock" (and approximate, word-count token figures), so for estimation
we fall back to the configured SUT model's rates — the number is illustrative
offline and exact against a real model.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import RunResult

# USD per 1M tokens, (input, output). Source: Anthropic pricing.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass
class PropertyMetrics:
    n_calls: int = 0
    n_errors: int = 0
    latency_mean_ms: float = 0.0
    latency_p95_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Empty -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def _price_for(model: str, fallback_model: str) -> tuple[float, float]:
    return PRICING.get(model) or PRICING.get(fallback_model) or (0.0, 0.0)


def estimate_cost(results: list[RunResult], fallback_model: str) -> float:
    """Estimated USD cost across results, using each call's model (or fallback)."""
    total = 0.0
    for r in results:
        in_price, out_price = _price_for(r.model, fallback_model)
        total += (r.usage.input_tokens * in_price + r.usage.output_tokens * out_price) / 1_000_000
    return total


def property_metrics(results: list[RunResult], fallback_model: str) -> PropertyMetrics:
    """Aggregate latency / token / cost metrics across all SUT calls."""
    latencies = [r.latency_ms for r in results]
    return PropertyMetrics(
        n_calls=len(results),
        n_errors=sum(1 for r in results if r.error),
        latency_mean_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_p95_ms=percentile(latencies, 95),
        total_input_tokens=sum(r.usage.input_tokens for r in results),
        total_output_tokens=sum(r.usage.output_tokens for r in results),
        estimated_cost_usd=estimate_cost(results, fallback_model),
    )
