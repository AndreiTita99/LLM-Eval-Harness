"""Orchestration: cases -> model outputs -> scores -> summary.

Phase 1 is intentionally minimal: load the golden dataset, send each case
through the SUT prompt once, parse the JSON output, and apply a single
hardcoded exact-match scorer on `category`. Later phases add the scorer
registry, N-repeats/variance, properties, baseline gating, and reporting.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Config
from .llm import LLMResponse, make_client
from .llm.judge import make_judge
from .models import EvalCase, RunResult, RunSummary, Usage
from .parsing import parse_json_output
from .scorers.registry import Registry, default_registry

# Re-exported for callers/tests that import it from the runner.
__all__ = ["load_cases", "load_prompt", "parse_json_output", "run"]


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load and validate golden cases from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of cases in {path}, got {type(raw).__name__}")
    return [EvalCase.model_validate(item) for item in raw]


def load_prompt(path: str | Path) -> str:
    """Load the SUT system prompt."""
    return Path(path).read_text(encoding="utf-8")


def _to_result(case: EvalCase, resp: LLMResponse) -> RunResult:
    return RunResult(
        case_id=case.id,
        raw_text=resp.text,
        parsed=parse_json_output(resp.text),
        latency_ms=resp.latency_ms,
        usage=Usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens),
        model=resp.model,
        error=resp.error,
    )


def run(
    cases: list[EvalCase],
    system_prompt: str,
    config: Config,
    registry: Registry | None = None,
) -> RunSummary:
    """Run every case once through the SUT and apply its declared scorers.

    Each case names the scorers that apply to it; the registry resolves those
    names to scorer instances. Names with no registry entry yet (e.g.
    `summary_judge` before phase 3) are recorded as skipped, not failed.
    """
    registry = registry or default_registry(judge=make_judge(config))
    client = make_client(config)
    summary = RunSummary(total_cases=len(cases))
    skipped: set[str] = set()

    for case in cases:
        resp = client.complete(system=system_prompt, user=case.input)
        result = _to_result(case, resp)
        summary.results.append(result)
        for scorer_name in case.scorers:
            scorer = registry.get(scorer_name)
            if scorer is None:
                skipped.add(scorer_name)
                continue
            summary.scores.append(scorer.score(case, result))

    summary.skipped_scorers = sorted(skipped)
    return summary
