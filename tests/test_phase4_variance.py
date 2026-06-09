"""Phase 4 tests — variance (N repeats / flaky detection) and property metrics."""

from __future__ import annotations

from src.config import Config
from src.metrics import PRICING, estimate_cost, percentile, property_metrics
from src.models import CaseScore, EvalCase, RunResult, RunSummary, Usage
from src.runner import run
from src.scorers.properties import FormatValid, NoRefusal


# --- variance / aggregation -------------------------------------------------

def _scores(case_id, scorer, outcomes):
    return [CaseScore(case_id=case_id, scorer=scorer, passed=p, repeat=i) for i, p in enumerate(outcomes)]


def test_case_score_stats_flags_flaky():
    summary = RunSummary(scores=_scores("c1", "category_exact", [True, True, False]))
    stat = summary.by_case_scorer()[0]
    assert stat.passed == 2 and stat.total == 3
    assert stat.is_flaky is True
    assert summary.flaky()[0].case_id == "c1"


def test_clean_pass_and_fail_are_not_flaky():
    summary = RunSummary(
        scores=_scores("c1", "s", [True, True, True]) + _scores("c2", "s", [False, False]),
    )
    assert summary.flaky() == []


def test_repeats_produce_one_result_per_repeat():
    config = Config(provider="mock", repeats=4)
    cases = [EvalCase(id="t1", input="My card was charged twice", expected={"category": "billing"},
                      scorers=["category_exact"])]
    summary = run(cases, system_prompt="p", config=config)
    assert len(summary.results) == 4
    assert sorted(r.repeat for r in summary.results) == [0, 1, 2, 3]


def test_mock_flakiness_is_reproducible_and_creates_variance():
    config = Config(provider="mock", repeats=8, mock_flakiness=0.5)
    cases = [EvalCase(id="t1", input="My card was charged twice for order 4471",
                      expected={"category": "billing"}, scorers=["category_exact"])]

    s1 = run(cases, system_prompt="p", config=config)
    s2 = run(cases, system_prompt="p", config=config)

    cat1 = [r.parsed["category"] for r in s1.results]
    cat2 = [r.parsed["category"] for r in s2.results]
    assert cat1 == cat2  # seeded -> reproducible run-to-run
    assert len(set(cat1)) > 1  # flakiness=0.5 over 8 repeats -> genuine variance

    stat = {s.scorer: s for s in s1.by_case_scorer()}["category_exact"]
    assert stat.is_flaky is True


# --- property scorers -------------------------------------------------------

def test_format_valid_scorer():
    case = EvalCase(id="c", input="x")
    assert FormatValid("format_valid").score(case, RunResult(case_id="c", raw_text="{}", parsed={})).passed is True
    bad = FormatValid("format_valid").score(case, RunResult(case_id="c", raw_text="oops", parsed=None))
    assert bad.passed is False


def test_no_refusal_scorer_detects_refusal():
    case = EvalCase(id="c", input="x")
    ok = NoRefusal("no_refusal").score(case, RunResult(case_id="c", raw_text='{"category":"billing"}', parsed={}))
    refused = NoRefusal("no_refusal").score(
        case, RunResult(case_id="c", raw_text="I'm unable to help with that request.", parsed=None)
    )
    assert ok.passed is True
    assert refused.passed is False


# --- property metrics -------------------------------------------------------

def test_percentile_nearest_rank():
    assert percentile([], 95) == 0.0
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([5], 50) == 5


def test_estimate_cost_uses_model_then_fallback():
    in_p, out_p = PRICING["claude-opus-4-8"]
    results = [
        RunResult(case_id="a", raw_text="", model="claude-opus-4-8", usage=Usage(input_tokens=1000, output_tokens=500)),
        RunResult(case_id="b", raw_text="", model="mock", usage=Usage(input_tokens=1000, output_tokens=500)),
    ]
    # mock has no price; falls back to the configured SUT model's rates -> same as the first row.
    expected_each = (1000 * in_p + 500 * out_p) / 1_000_000
    assert estimate_cost(results, fallback_model="claude-opus-4-8") == expected_each * 2


def test_property_metrics_aggregate():
    results = [
        RunResult(case_id="a", raw_text="", model="mock", latency_ms=10, usage=Usage(input_tokens=5, output_tokens=3)),
        RunResult(case_id="b", raw_text="", model="mock", latency_ms=30, usage=Usage(input_tokens=7, output_tokens=2), error="boom"),
    ]
    m = property_metrics(results, fallback_model="claude-haiku-4-5")
    assert m.n_calls == 2
    assert m.n_errors == 1
    assert m.latency_mean_ms == 20
    assert m.total_input_tokens == 12
    assert m.total_output_tokens == 5
