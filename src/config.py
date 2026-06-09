"""Typed configuration for the eval harness.

Everything that controls a run lives here: which models to use, how many times to
repeat each case, and (later phases) regression tolerances. Config is read from
environment variables with sensible defaults so the harness is runnable with zero
setup against the built-in mock provider.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["anthropic", "mock"]


def _default_provider() -> Provider:
    """Use the real API when a key is present, otherwise the offline mock.

    This makes `eval run` work out of the box for a reviewer who hasn't set up an
    API key, while still exercising the full pipeline against a real model when
    credentials are available.
    """
    explicit = os.getenv("EVAL_PROVIDER")
    if explicit in ("anthropic", "mock"):
        return explicit  # type: ignore[return-value]
    return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "mock"


class Config(BaseModel):
    """Run configuration. Model ids are config, never hardcoded in call sites."""

    # --- Provider / models ---
    provider: Provider = Field(default_factory=_default_provider)

    # The model whose prompt is under test. Configurable so a model swap is a
    # one-line change and the judge can differ from the SUT.
    sut_model: str = Field(default="claude-opus-4-8")

    # The judge runs on a cheaper, different model than the SUT to reduce
    # self-preference bias and cost.
    judge_model: str = Field(default="claude-haiku-4-5")

    max_tokens: int = Field(default=1024)
    # The judge emits a short {score, reasoning}; it needs little headroom.
    judge_max_tokens: int = Field(default=512)
    # Minimum rubric score (1..3) the judge must give for a summary to pass.
    judge_pass_threshold: int = Field(default=2, ge=1, le=3)

    # Sampling temperature for the SUT. Left unset by default because the most
    # capable models (Opus 4.8/4.7) no longer expose temperature at all — so we
    # lean on N-repeats + pass-rate to characterise non-determinism rather than
    # pinning a temperature. Set this only for models that accept it.
    sut_temperature: float | None = Field(default=None)

    # --- Variance handling (phase 4) ---
    repeats: int = Field(default=3, ge=1)

    @classmethod
    def from_env(cls) -> "Config":
        """Build config from environment variables, falling back to defaults."""
        overrides: dict[str, object] = {}
        if model := os.getenv("EVAL_SUT_MODEL"):
            overrides["sut_model"] = model
        if model := os.getenv("EVAL_JUDGE_MODEL"):
            overrides["judge_model"] = model
        if repeats := os.getenv("EVAL_REPEATS"):
            overrides["repeats"] = int(repeats)
        if temp := os.getenv("EVAL_SUT_TEMPERATURE"):
            overrides["sut_temperature"] = float(temp)
        if threshold := os.getenv("EVAL_JUDGE_THRESHOLD"):
            overrides["judge_pass_threshold"] = int(threshold)
        return cls(**overrides)
