"""Property scorers — intrinsic checks that apply to *every* model call.

Unlike structural and judge scorers (which need per-case expected values and are
declared per case), property checks are universal: any call can be malformed or a
refusal, so the runner applies these to every result automatically. Latency and
token cost are also properties, but they're continuous metrics rather than
pass/fail checks — see `src/metrics.py`.
"""

from __future__ import annotations

from ..models import EvalCase, RunResult
from .base import Scorer

# Phrases that signal the model declined rather than answering. Deliberately
# conservative — a triage assistant should essentially never refuse, so any of
# these in the raw output is worth flagging.
_REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i am unable to",
    "i can't provide",
    "i cannot provide",
    "as an ai",
    "i'm not able to",
)


class FormatValid(Scorer):
    """Pass iff the model returned parseable JSON (format health)."""

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        if result.error:
            return False, 0.0, f"call error: {result.error}"
        passed = result.parsed is not None
        return passed, 1.0 if passed else 0.0, "parseable JSON" if passed else "unparseable output"


class NoRefusal(Scorer):
    """Pass iff the model did not refuse / decline to answer."""

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        text = result.raw_text.lower()
        hit = next((m for m in _REFUSAL_MARKERS if m in text), None)
        passed = hit is None
        return passed, 1.0 if passed else 0.0, "no refusal" if passed else f"refusal marker: {hit!r}"


def property_scorers() -> list[Scorer]:
    """Universal property scorers applied to every result, regardless of case."""
    return [FormatValid("format_valid"), NoRefusal("no_refusal")]
