"""Phase 2 tests — structural scorers, registry, and aggregation (no LLM)."""

from __future__ import annotations

import pytest

from src.config import Config
from src.models import EvalCase, RunResult
from src.runner import run
from src.scorers.registry import default_registry
from src.scorers.structural import Contains, EnumValid, ExactMatch, SchemaValid


def _result(parsed):
    return RunResult(case_id="c", raw_text="", parsed=parsed)


def test_exact_match_pass_and_fail():
    case = EvalCase(id="c", input="x", expected={"category": "billing"})
    s = ExactMatch("category_exact", "category")
    assert s.score(case, _result({"category": "billing"})).passed is True
    assert s.score(case, _result({"category": "technical"})).passed is False


def test_exact_match_handles_unparseable():
    case = EvalCase(id="c", input="x", expected={"category": "billing"})
    score = ExactMatch("category_exact", "category").score(case, _result(None))
    assert score.passed is False
    assert "parseable" in score.detail


def test_enum_valid_checks_membership_not_correctness():
    case = EvalCase(id="c", input="x", expected={"urgency": "high"})
    s = EnumValid("urgency_schema", "urgency", ["low", "medium", "high"])
    # 'low' is a valid enum value even though expected is 'high' — enum validity
    # is a format check, not a correctness check.
    assert s.score(case, _result({"urgency": "low"})).passed is True
    assert s.score(case, _result({"urgency": "screaming"})).passed is False


def test_contains_all_and_any():
    case = EvalCase(id="c", input="x")
    all_s = Contains("c_all", "summary", ["charged", "twice"], mode="all")
    any_s = Contains("c_any", "summary", ["charged", "missing"], mode="any")
    r = _result({"summary": "Customer was CHARGED twice"})
    assert all_s.score(case, r).passed is True
    assert any_s.score(case, r).passed is True
    assert all_s.score(case, _result({"summary": "charged once"})).passed is False


def test_contains_rejects_bad_mode():
    with pytest.raises(ValueError):
        Contains("c", "f", ["x"], mode="most")


def test_schema_valid_reports_each_violation():
    spec = {
        "category": {"type": str, "enum": ["billing"], "required": True},
        "summary": {"type": str, "required": True, "non_empty": True},
    }
    s = SchemaValid("response_schema", spec)
    assert s.score(EvalCase(id="c", input="x"), _result({"category": "billing", "summary": "hi"})).passed is True

    bad = s.score(EvalCase(id="c", input="x"), _result({"category": "wrong", "summary": ""}))
    assert bad.passed is False
    assert "enum" in bad.detail and "empty" in bad.detail


def test_registry_resolves_known_and_misses_unknown():
    reg = default_registry()  # no judge supplied
    assert reg.get("category_exact") is not None
    assert reg.get("summary_judge") is None  # judge only registered when supplied


def test_run_skips_truly_unregistered_scorers_and_aggregates():
    config = Config(provider="mock", repeats=1)
    cases = [
        EvalCase(
            id="t1",
            input="My card was charged twice for order 4471",
            expected={"category": "billing", "urgency": "high"},
            scorers=["category_exact", "urgency_schema", "response_schema", "nonexistent_scorer"],
        ),
    ]
    summary = run(cases, system_prompt="triage", config=config)

    # An unknown scorer is skipped; the structural scorers and the universal
    # property scorers still run.
    assert summary.skipped_scorers == ["nonexistent_scorer"]
    assert {s.scorer for s in summary.scores} == {
        "category_exact", "urgency_schema", "response_schema", "format_valid", "no_refusal",
    }

    by_scorer = {s.scorer: s for s in summary.by_scorer()}
    assert by_scorer["response_schema"].pass_rate == 1.0
