"""Anthropic SDK wrapper plus an offline mock, behind one small interface.

The harness is deliberately single-provider (Anthropic) but talks to it through
a narrow `complete(system, user) -> LLMResponse` surface. That keeps call sites
provider-agnostic and lets the mock stand in for CI and zero-setup demos.

A batch path (Message Batches API, ~50% cheaper) is noted for phase 6; the sync
path here is what the PR-gate subset uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Config


@dataclass
class LLMResponse:
    """Normalised result of one model call."""

    text: str
    model: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class AnthropicClient:
    """Thin wrapper over the official `anthropic` SDK (sync path)."""

    config: Config
    _client: object = field(default=None, repr=False)

    def _ensure_client(self) -> object:
        if self._client is None:
            import anthropic  # imported lazily so the mock path needs no SDK

            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        model = model or self.config.sut_model
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        # Only send temperature when explicitly configured — Opus 4.8/4.7 reject
        # it, so omitting keeps the harness model-agnostic.
        if self.config.sut_temperature is not None:
            kwargs["temperature"] = self.config.sut_temperature

        start = time.perf_counter()
        try:
            msg = client.messages.create(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:  # surfaced as a failed result, not a crash
            return LLMResponse(
                text="",
                model=model,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return LLMResponse(
            text=text,
            model=getattr(msg, "model", model),
            latency_ms=latency_ms,
            input_tokens=getattr(msg.usage, "input_tokens", 0),
            output_tokens=getattr(msg.usage, "output_tokens", 0),
        )


@dataclass
class MockClient:
    """Deterministic offline stand-in.

    Produces plausible triage JSON from simple keyword heuristics so the full
    pipeline (and pass/fail scoring) runs with no API key. Intentionally
    imperfect — some cases will mis-classify, which is exactly what makes the
    scoring output meaningful in a demo.
    """

    config: Config

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import json

        start = time.perf_counter()
        category = _mock_category(user)
        urgency = _mock_urgency(user)
        summary = _mock_summary(user)
        payload = {"category": category, "urgency": urgency, "summary": summary}
        text = json.dumps(payload)
        # Small synthetic latency so latency metrics aren't all zero in demos.
        latency_ms = (time.perf_counter() - start) * 1000 + 5.0
        return LLMResponse(
            text=text,
            model="mock",
            latency_ms=latency_ms,
            input_tokens=len(user.split()),
            output_tokens=len(text.split()),
        )


def _mock_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("charg", "refund", "invoice", "payment", "card", "bill")):
        return "billing"
    if any(k in t for k in ("password", "login", "log in", "error", "bug", "crash", "broken", "not working")):
        return "technical"
    if any(k in t for k in ("cancel", "subscription", "account", "downgrade", "upgrade plan")):
        return "account"
    if any(k in t for k in ("ship", "deliver", "track", "package", "order", "arrive")):
        return "shipping"
    return "general"


def _mock_urgency(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("urgent", "immediately", "asap", "twice", "can't access", "cannot access", "down", "locked out")):
        return "high"
    if any(k in t for k in ("soon", "when", "still waiting", "delayed")):
        return "medium"
    return "low"


def _mock_summary(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:60] + ("…" if len(collapsed) > 60 else "")


def make_client(config: Config):
    """Return the client matching the configured provider."""
    if config.provider == "mock":
        return MockClient(config)
    return AnthropicClient(config)
