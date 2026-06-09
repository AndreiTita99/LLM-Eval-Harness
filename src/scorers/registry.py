"""Scorer registry — maps scorer names to scorer instances.

Cases in the golden dataset declare which scorers apply by name (e.g.
`category_exact`). The registry resolves those names to concrete scorers. This
is what turns the harness from "a script with one scorer" into a framework:
adding a metric is registering one entry, and the dataset opts cases in.

The SUT-specific configuration (allowed categories/urgencies, the response
schema) lives here — the scorer primitives themselves stay generic.
"""

from __future__ import annotations

from .base import Scorer
from .structural import EnumValid, ExactMatch, SchemaValid

# --- Triage SUT vocabulary ---
ALLOWED_CATEGORIES = ["billing", "technical", "account", "shipping", "general"]
ALLOWED_URGENCY = ["low", "medium", "high"]

TRIAGE_SCHEMA = {
    "category": {"type": str, "enum": ALLOWED_CATEGORIES, "required": True},
    "urgency": {"type": str, "enum": ALLOWED_URGENCY, "required": True},
    "summary": {"type": str, "required": True, "non_empty": True},
}


class Registry:
    """A name -> Scorer mapping with resolve/skip semantics."""

    def __init__(self) -> None:
        self._scorers: dict[str, Scorer] = {}

    def register(self, scorer: Scorer) -> None:
        if scorer.name in self._scorers:
            raise ValueError(f"scorer {scorer.name!r} already registered")
        self._scorers[scorer.name] = scorer

    def get(self, name: str) -> Scorer | None:
        return self._scorers.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._scorers

    def names(self) -> list[str]:
        return sorted(self._scorers)


def default_registry() -> Registry:
    """Registry with the structural scorers wired up for the triage SUT.

    `summary_judge` is intentionally absent until phase 3 (LLM-as-judge); cases
    that declare it have it skipped, not failed.
    """
    reg = Registry()
    reg.register(ExactMatch("category_exact", field="category"))
    reg.register(EnumValid("urgency_schema", field="urgency", allowed=ALLOWED_URGENCY))
    reg.register(SchemaValid("response_schema", spec=TRIAGE_SCHEMA))
    return reg
