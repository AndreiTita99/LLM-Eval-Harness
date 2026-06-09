"""Pydantic data models shared across the harness.

These are the typed contracts that flow through the pipeline:

    EvalCase   -> a golden input case loaded from YAML
    RunResult  -> the output of one model call (output + latency + cost)
    CaseScore  -> a single scorer's verdict on a RunResult (phase 2+)
    RunSummary -> aggregate metrics across the whole run (phase 4+)

Phase 1 only populates EvalCase and RunResult; the rest are defined here so the
shape of the system is clear from the start.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One golden case: an input, the values we expect, and which scorers apply."""

    id: str
    input: str
    expected: dict[str, Any] = Field(default_factory=dict)
    # Names of scorers to apply, resolved against the registry in phase 2.
    scorers: list[str] = Field(default_factory=list)
    # Marks cases held out from prompt iteration (the "don't test on your
    # training data" split). Surfaced in reporting; not used for gating logic.
    held_out: bool = False


class Usage(BaseModel):
    """Token usage for a single model call."""

    input_tokens: int = 0
    output_tokens: int = 0


class RunResult(BaseModel):
    """The result of sending one case through the SUT prompt once."""

    case_id: str
    # Raw model text exactly as returned.
    raw_text: str
    # Parsed structured output (e.g. {category, urgency, summary}), or None if
    # the model didn't return valid JSON. Format-validity is itself a metric.
    parsed: dict[str, Any] | None = None
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    # Index of the repeat this result came from (0-based); relevant in phase 4.
    repeat: int = 0
    error: str | None = None


class CaseScore(BaseModel):
    """A single scorer's verdict on one RunResult."""

    case_id: str
    scorer: str
    passed: bool
    score: float = 0.0  # normalised 0..1
    detail: str = ""
    # Which repeat (0-based) this verdict came from — variance handling (phase 4).
    repeat: int = 0


class ScorerStats(BaseModel):
    """Pass-rate for a single scorer across all cases it was applied to."""

    scorer: str
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class CaseScoreStats(BaseModel):
    """Pass-rate for one (case, scorer) pair across N repeats — variance handling.

    A case that passes 3/5 is a different signal than one that passes 5/5;
    `is_flaky` flags the strictly-in-between case so it isn't read as a clean pass
    or a clean fail.
    """

    case_id: str
    scorer: str
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def is_flaky(self) -> bool:
        return self.total > 1 and 0 < self.passed < self.total


class RunSummary(BaseModel):
    """Aggregate metrics for a whole run (filled out in later phases)."""

    total_cases: int = 0
    results: list[RunResult] = Field(default_factory=list)
    scores: list[CaseScore] = Field(default_factory=list)
    # Declared scorers that no registry entry exists for yet (e.g. summary_judge
    # before phase 3). Skipped, not failed — surfaced so they're not silent.
    skipped_scorers: list[str] = Field(default_factory=list)

    def by_scorer(self) -> list[ScorerStats]:
        """Per-scorer pass counts, ordered by scorer name."""
        agg: dict[str, list[int]] = {}
        for s in self.scores:
            bucket = agg.setdefault(s.scorer, [0, 0])
            bucket[0] += int(s.passed)
            bucket[1] += 1
        return [ScorerStats(scorer=name, passed=p, total=t) for name, (p, t) in sorted(agg.items())]

    def by_case_scorer(self) -> list[CaseScoreStats]:
        """Per (case, scorer) pass-rate across repeats, ordered case then scorer."""
        agg: dict[tuple[str, str], list[int]] = {}
        for s in self.scores:
            bucket = agg.setdefault((s.case_id, s.scorer), [0, 0])
            bucket[0] += int(s.passed)
            bucket[1] += 1
        return [
            CaseScoreStats(case_id=cid, scorer=name, passed=p, total=t)
            for (cid, name), (p, t) in sorted(agg.items())
        ]

    def flaky(self) -> list[CaseScoreStats]:
        """(case, scorer) pairs that neither always passed nor always failed."""
        return [s for s in self.by_case_scorer() if s.is_flaky]

    def overall(self) -> ScorerStats:
        """Pass count across every (case, scorer, repeat) verdict in the run."""
        passed = sum(1 for s in self.scores if s.passed)
        return ScorerStats(scorer="overall", passed=passed, total=len(self.scores))
