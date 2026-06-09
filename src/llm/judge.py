"""LLM-as-judge: grade a free-text summary against an explicit rubric.

This is the hard part of any eval harness, and the part most people get wrong.
The mitigations here are deliberate and worth being able to explain:

  - **Explicit rubric, not a vague 1-10.** Three levels, each defined. A judge
    grading against defined anchors is far more stable than "rate 1-10".
  - **Verbosity-bias guard.** The prompt tells the judge that a concise correct
    summary scores as high as a verbose one — length is not quality.
  - **Self-preference guard.** The judge runs on a *different, cheaper* model
    (`judge_model`, default Haiku) than the SUT (default Opus), so it isn't
    grading its own house style.
  - **Position bias** is a pairwise-comparison problem; this judge grades a
    single output pointwise, so it doesn't apply here (noted, not ignored).

The judge is itself validated against human labels — see `src/validation.py`
and `eval judge-validate`. A judge you haven't validated is just vibes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import Config
from ..parsing import parse_json_output

ALLOWED_SCORES = (1, 2, 3)

RUBRIC_LEVELS = {
    3: "Accurately and concisely captures the customer's core issue. No invented details.",
    2: "Captures the gist but is vague, slightly inaccurate, or includes minor extraneous detail.",
    1: "Misrepresents or omits the core issue, is empty, or hallucinates details not in the ticket.",
}

JUDGE_SYSTEM = """You are a strict, impartial evaluator. You grade the quality of a one-line \
summary of a customer support ticket, on a 1-3 scale, against this rubric:

  3 — {three}
  2 — {two}
  1 — {one}

Grade ONLY against the rubric. Apply these rules to stay unbiased:
  - Length is not quality: a short, correct summary scores exactly as high as a
    long one. Do not reward verbosity.
  - Ignore writing style, tone, grammar, and formatting. Judge only whether the
    summary accurately and concisely captures the customer's core issue.
  - Penalise any detail in the summary that is not supported by the ticket.

Respond with ONLY a JSON object, no preamble or code fences:
  {{"score": <1|2|3>, "reasoning": "<one short sentence>"}}""".format(
    three=RUBRIC_LEVELS[3], two=RUBRIC_LEVELS[2], one=RUBRIC_LEVELS[1]
)


def build_judge_user(ticket: str, summary: str) -> str:
    """The per-grade user turn: the ticket and the candidate summary."""
    return (
        f'Customer ticket:\n"""{ticket}"""\n\n'
        f'Candidate one-line summary:\n"""{summary}"""\n\n'
        "Grade the summary against the rubric."
    )


@dataclass
class JudgeResult:
    """Outcome of grading one summary."""

    score: int | None  # 1..3, or None if the judge output couldn't be parsed
    reasoning: str
    passed: bool
    latency_ms: float = 0.0
    error: str | None = None


class Judge(Protocol):
    def grade(self, ticket: str, summary: str) -> JudgeResult: ...


def _verdict(score: int | None, reasoning: str, threshold: int, latency_ms: float, error: str | None = None) -> JudgeResult:
    passed = score is not None and score >= threshold
    return JudgeResult(score=score, reasoning=reasoning, passed=passed, latency_ms=latency_ms, error=error)


class AnthropicJudge:
    """Grades via the Anthropic API on the configured (cheaper) judge model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # Reuse the SUT client wrapper, overriding the model per call.
        from .client import AnthropicClient

        self._client = AnthropicClient(config)

    def grade(self, ticket: str, summary: str) -> JudgeResult:
        resp = self._client.complete(
            JUDGE_SYSTEM,
            build_judge_user(ticket, summary),
            model=self.config.judge_model,
            max_tokens=self.config.judge_max_tokens,
        )
        if resp.error:
            return _verdict(None, "", self.config.judge_pass_threshold, resp.latency_ms, resp.error)

        data = parse_json_output(resp.text)
        if not data or data.get("score") not in ALLOWED_SCORES:
            return _verdict(
                None,
                "",
                self.config.judge_pass_threshold,
                resp.latency_ms,
                error=f"unparseable judge output: {resp.text[:80]!r}",
            )
        return _verdict(
            int(data["score"]),
            str(data.get("reasoning", "")),
            self.config.judge_pass_threshold,
            resp.latency_ms,
        )


# --- Offline mock judge -----------------------------------------------------

import re

_STOPWORDS = {
    "customer", "about", "their", "there", "would", "could", "please",
    "after", "before", "still", "that", "this", "with", "into", "from",
}


def _content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


def mock_summary_score(ticket: str, summary: str) -> int:
    """Heuristic stand-in for a real judge — deliberately imperfect.

    Scores on keyword overlap with the ticket and conciseness. It intentionally
    over-penalises long summaries (a crude verbosity bias) and can't detect
    hallucinated detail — exactly the kinds of judge error the human-agreement
    check is designed to expose.
    """
    if not summary.strip():
        return 1
    overlap = len(_content_tokens(summary) & _content_tokens(ticket))
    if overlap == 0:
        return 1
    if overlap >= 2 and len(summary.split()) <= 20:
        return 3
    return 2


class MockJudge:
    """Deterministic offline judge so the pipeline runs with no API key."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def grade(self, ticket: str, summary: str) -> JudgeResult:
        score = mock_summary_score(ticket, summary)
        return _verdict(score, f"mock heuristic score {score}", self.config.judge_pass_threshold, latency_ms=2.0)


def make_judge(config: Config) -> Judge:
    """Return the judge matching the configured provider."""
    if config.provider == "mock":
        return MockJudge(config)
    return AnthropicJudge(config)
