"""Phase 3 tests — LLM-as-judge scorer and human-agreement validation (mock judge)."""

from __future__ import annotations

from src.config import Config
from src.llm.judge import MockJudge, make_judge, mock_summary_score
from src.models import EvalCase, RunResult
from src.scorers.judge import SummaryJudge
from src.validation import _cohen_kappa_binary, validate_judge


def test_mock_summary_score_levels():
    ticket = "My card was charged twice for order 4471."
    assert mock_summary_score(ticket, "") == 1  # empty
    assert mock_summary_score(ticket, "Completely unrelated text here") == 1  # no overlap
    assert mock_summary_score(ticket, "Customer charged twice on order.") == 3  # concise + overlap
    verbose = "The customer reports they were charged twice " * 6  # long, has overlap
    assert mock_summary_score(ticket, verbose) == 2  # length-penalised


def test_summary_judge_scorer_pass_and_fail():
    judge = MockJudge(Config(provider="mock"))
    scorer = SummaryJudge("summary_judge", judge=judge, field="summary")
    case = EvalCase(id="c", input="My card was charged twice for order 4471.")

    good = scorer.score(case, RunResult(case_id="c", raw_text="", parsed={"summary": "Charged twice on order."}))
    bad = scorer.score(case, RunResult(case_id="c", raw_text="", parsed={"summary": ""}))
    assert good.passed is True
    assert bad.passed is False


def test_summary_judge_handles_unparseable_output():
    judge = MockJudge(Config(provider="mock"))
    scorer = SummaryJudge("summary_judge", judge=judge)
    score = scorer.score(EvalCase(id="c", input="x"), RunResult(case_id="c", raw_text="", parsed=None))
    assert score.passed is False
    assert "parseable" in score.detail


def test_make_judge_selects_mock_offline():
    assert isinstance(make_judge(Config(provider="mock")), MockJudge)


def test_cohen_kappa_perfect_and_chance():
    assert _cohen_kappa_binary([True, False, True], [True, False, True]) == 1.0
    # All agree but one rater is constant → no better than chance.
    assert _cohen_kappa_binary([True, True, True], [True, True, True]) == 1.0


def test_validate_judge_reports_agreement(tmp_path):
    labeled = [
        {"id": "a", "input": "card charged twice on order", "summary": "Charged twice on order.", "human_score": 3},
        {"id": "b", "input": "cancel my subscription", "summary": "", "human_score": 1},
    ]
    report = validate_judge(MockJudge(Config(provider="mock")), labeled, threshold=2)
    assert report.n == 2
    # Both should agree (good->3 pass, empty->1 fail), so perfect agreement here.
    assert report.exact_agreement == 1.0
    assert report.passfail_agreement == 1.0
