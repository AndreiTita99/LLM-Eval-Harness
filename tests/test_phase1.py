"""Phase 1 smoke tests — no live LLM calls (mock provider only)."""

from __future__ import annotations

from src.config import Config
from src.models import EvalCase, RunResult
from src.runner import parse_json_output, run, score_category_exact


def test_parse_json_output_plain():
    assert parse_json_output('{"category": "billing"}') == {"category": "billing"}


def test_parse_json_output_strips_code_fence():
    text = '```json\n{"category": "technical"}\n```'
    assert parse_json_output(text) == {"category": "technical"}


def test_parse_json_output_returns_none_on_garbage():
    assert parse_json_output("not json at all") is None


def test_score_category_exact_pass_and_fail():
    case = EvalCase(id="c1", input="x", expected={"category": "billing"})
    hit = RunResult(case_id="c1", raw_text="", parsed={"category": "billing"})
    miss = RunResult(case_id="c1", raw_text="", parsed={"category": "technical"})

    assert score_category_exact(case, hit).passed is True
    assert score_category_exact(case, miss).passed is False


def test_run_end_to_end_with_mock():
    config = Config(provider="mock")
    cases = [
        EvalCase(id="t1", input="My card was charged twice", expected={"category": "billing"}),
        EvalCase(id="t2", input="How do I crash-fix this bug?", expected={"category": "technical"}),
    ]
    summary = run(cases, system_prompt="triage", config=config)

    assert summary.total_cases == 2
    assert len(summary.results) == 2
    assert len(summary.scores) == 2
    # The mock parses to valid JSON for every case.
    assert all(r.parsed is not None for r in summary.results)
