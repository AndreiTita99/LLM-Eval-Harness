"""Judge validation: measure the judge's agreement with human labels.

"A judge you haven't validated is just vibes." This module runs the judge over a
small hand-labelled set and reports how often it agrees with the human grader:

  - **Exact agreement** — judge score == human score (1..3).
  - **Pass/fail agreement** — both sides agree on pass vs fail at the threshold.
    Usually the metric that matters for gating, since the gate is binary.
  - **Cohen's kappa** — agreement corrected for chance, on the pass/fail labels.
    Rough reading: <0 worse than chance, 0.0-0.2 slight, 0.2-0.4 fair,
    0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 almost perfect.

If the judge disagrees with humans, the gate is built on sand — so this number
is reported alongside the eval, not buried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .llm.judge import Judge


@dataclass
class JudgeComparison:
    id: str
    human_score: int
    judge_score: int | None
    agree_exact: bool
    agree_passfail: bool
    reasoning: str = ""


@dataclass
class ValidationReport:
    threshold: int
    comparisons: list[JudgeComparison] = field(default_factory=list)
    exact_agreement: float = 0.0
    passfail_agreement: float = 0.0
    kappa: float = 0.0

    @property
    def n(self) -> int:
        return len(self.comparisons)


def load_labeled(path: str | Path) -> list[dict]:
    """Load a hand-labelled judge dataset (id, input, summary, human_score)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of labelled examples in {path}")
    return raw


def _cohen_kappa_binary(human: list[bool], judge: list[bool]) -> float:
    """Cohen's kappa for two binary raters. Returns 0.0 for degenerate inputs."""
    n = len(human)
    if n == 0:
        return 0.0
    po = sum(h == j for h, j in zip(human, judge)) / n
    p_human_yes = sum(human) / n
    p_judge_yes = sum(judge) / n
    pe = p_human_yes * p_judge_yes + (1 - p_human_yes) * (1 - p_judge_yes)
    if pe >= 1.0:  # both raters unanimous and identical → perfect, undefined chance
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def validate_judge(judge: Judge, labeled: list[dict], threshold: int) -> ValidationReport:
    """Run the judge over labelled examples and compute agreement metrics."""
    report = ValidationReport(threshold=threshold)
    human_pass: list[bool] = []
    judge_pass: list[bool] = []

    for ex in labeled:
        human = int(ex["human_score"])
        verdict = judge.grade(ex["input"], ex.get("summary", ""))
        js = verdict.score
        h_pass = human >= threshold
        j_pass = js is not None and js >= threshold
        report.comparisons.append(
            JudgeComparison(
                id=str(ex.get("id", "")),
                human_score=human,
                judge_score=js,
                agree_exact=(js == human),
                agree_passfail=(h_pass == j_pass),
                reasoning=verdict.reasoning,
            )
        )
        human_pass.append(h_pass)
        judge_pass.append(j_pass)

    n = report.n
    if n:
        report.exact_agreement = sum(c.agree_exact for c in report.comparisons) / n
        report.passfail_agreement = sum(c.agree_passfail for c in report.comparisons) / n
        report.kappa = _cohen_kappa_binary(human_pass, judge_pass)
    return report
