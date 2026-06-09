"""Scorer interface shared by all scorer families.

A scorer takes a case and one model result and returns a CaseScore. Subclasses
implement `_evaluate` and the base wraps the verdict into a CaseScore tagged with
the scorer's registered name, so every family (structural, judge, property) has
the same shape.
"""

from __future__ import annotations

from ..models import CaseScore, EvalCase, RunResult


class Scorer:
    """Base class: holds a registered name and emits a tagged CaseScore."""

    def __init__(self, name: str) -> None:
        self.name = name

    def score(self, case: EvalCase, result: RunResult) -> CaseScore:
        passed, score, detail = self._evaluate(case, result)
        return CaseScore(
            case_id=case.id,
            scorer=self.name,
            passed=passed,
            score=score,
            detail=detail,
            repeat=result.repeat,
        )

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        raise NotImplementedError
