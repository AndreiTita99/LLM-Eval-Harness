"""Shared parsing helpers.

Kept in its own module so both the runner and the judge can use it without a
circular import (runner -> scorers -> llm.judge -> parsing).
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_output(text: str) -> dict | None:
    """Best-effort parse of a model's JSON response.

    Tolerates ```json fences and surrounding whitespace. Returns None on failure
    — format validity is a first-class metric, not a crash.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
