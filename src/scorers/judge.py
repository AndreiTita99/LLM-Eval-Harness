"""LLM-as-judge scorer — adapts a Judge to the Scorer interface.

Pulls the free-text field (e.g. `summary`) out of the parsed model output, asks
the judge to grade it against the rubric, and turns the judge's verdict into a
CaseScore. The grading logic and bias guards live in `src/llm/judge.py`; this is
just the adapter.
"""

from __future__ import annotations

from ..llm.judge import Judge
from ..models import EvalCase, RunResult
from .base import Scorer


class SummaryJudge(Scorer):
    """Grade the parsed `field` of a result with an LLM judge."""

    def __init__(self, name: str, judge: Judge, field: str = "summary") -> None:
        super().__init__(name)
        self.judge = judge
        self.field = field

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        if result.parsed is None:
            return False, 0.0, "no parseable output"
        summary = str(result.parsed.get(self.field, ""))
        verdict = self.judge.grade(case.input, summary)
        if verdict.error:
            return False, 0.0, f"judge error: {verdict.error}"
        # Normalise the 1..3 rubric score onto 0..1 for aggregation.
        norm = (verdict.score or 0) / 3.0
        mark = "pass" if verdict.passed else "fail"
        detail = f"score={verdict.score} ({mark}): {verdict.reasoning[:60]}"
        return verdict.passed, norm, detail
