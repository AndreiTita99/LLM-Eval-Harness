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
    """A single scorer's verdict on a RunResult (phase 2+)."""

    case_id: str
    scorer: str
    passed: bool
    score: float = 0.0  # normalised 0..1
    detail: str = ""


class RunSummary(BaseModel):
    """Aggregate metrics for a whole run (filled out in later phases)."""

    total_cases: int = 0
    results: list[RunResult] = Field(default_factory=list)
    scores: list[CaseScore] = Field(default_factory=list)
