"""Structural scorers — deterministic, cheap, and the backbone of the harness.

These never call an LLM. They check the shape and content of the parsed output:
exact match on a field, enum/schema validity, substring presence, and whole-
response schema validation. They're parameterised primitives, so the same code
serves any prompt — the SUT-specific instances are wired up in `registry.py`.

All scorers treat an unparseable response (`result.parsed is None`) as a fail
with a clear detail string rather than raising.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..models import EvalCase, RunResult
from .base import Scorer


class ExactMatch(Scorer):
    """Pass iff the parsed `field` exactly equals `expected[field]`."""

    def __init__(self, name: str, field: str) -> None:
        super().__init__(name)
        self.field = field

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        if result.parsed is None:
            return False, 0.0, "no parseable output"
        expected = case.expected.get(self.field)
        actual = result.parsed.get(self.field)
        passed = expected is not None and actual == expected
        return passed, 1.0 if passed else 0.0, f"expected={expected!r} actual={actual!r}"


class EnumValid(Scorer):
    """Pass iff the parsed `field` is one of the allowed values (enum validity).

    This is a *format* check (is the value in the allowed set?), distinct from
    ExactMatch which checks correctness against the expected label.
    """

    def __init__(self, name: str, field: str, allowed: Iterable[str]) -> None:
        super().__init__(name)
        self.field = field
        self.allowed = list(allowed)

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        if result.parsed is None:
            return False, 0.0, "no parseable output"
        actual = result.parsed.get(self.field)
        passed = actual in self.allowed
        detail = f"{self.field}={actual!r} allowed={self.allowed}"
        return passed, 1.0 if passed else 0.0, detail


class Contains(Scorer):
    """Pass iff the parsed `field` contains the given substring(s).

    `mode='all'` requires every substring; `mode='any'` requires at least one.
    Case-insensitive. A general-purpose primitive (not used by the triage SUT by
    default, but part of the structural family).
    """

    def __init__(
        self,
        name: str,
        field: str,
        substrings: Iterable[str],
        mode: str = "all",
    ) -> None:
        super().__init__(name)
        self.field = field
        self.substrings = [s.lower() for s in substrings]
        if mode not in ("all", "any"):
            raise ValueError("mode must be 'all' or 'any'")
        self.mode = mode

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        if result.parsed is None:
            return False, 0.0, "no parseable output"
        value = str(result.parsed.get(self.field, "")).lower()
        hits = [s for s in self.substrings if s in value]
        passed = len(hits) == len(self.substrings) if self.mode == "all" else len(hits) > 0
        return passed, 1.0 if passed else 0.0, f"matched {len(hits)}/{len(self.substrings)} ({self.mode})"


# A field spec for SchemaValid:
#   {"field": {"type": <pytype>, "enum": [...], "required": bool, "non_empty": bool}}
FieldSpec = dict[str, dict[str, Any]]


class SchemaValid(Scorer):
    """Validate the whole parsed response against a lightweight field schema.

    Checks required-key presence, Python type per field, optional enum
    membership, and an optional non-empty constraint. Dependency-free so the
    harness stays portable; the same idea pydantic would give us, scoped to one
    scorer rather than coupling the SUT shape into the core.
    """

    def __init__(self, name: str, spec: FieldSpec) -> None:
        super().__init__(name)
        self.spec = spec

    def _evaluate(self, case: EvalCase, result: RunResult) -> tuple[bool, float, str]:
        if result.parsed is None:
            return False, 0.0, "no parseable output"
        errors: list[str] = []
        for field, rules in self.spec.items():
            present = field in result.parsed
            if not present:
                if rules.get("required", False):
                    errors.append(f"missing '{field}'")
                continue
            value = result.parsed[field]
            expected_type = rules.get("type")
            if expected_type is not None and not isinstance(value, expected_type):
                errors.append(f"'{field}' wrong type ({type(value).__name__})")
                continue
            if "enum" in rules and value not in rules["enum"]:
                errors.append(f"'{field}'={value!r} not in enum")
            if rules.get("non_empty") and not str(value).strip():
                errors.append(f"'{field}' is empty")
        passed = not errors
        return passed, 1.0 if passed else 0.0, "valid" if passed else "; ".join(errors)
