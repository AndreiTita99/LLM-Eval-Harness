"""LLM provider abstraction: one provider behind one interface."""

from .client import LLMResponse, make_client

__all__ = ["LLMResponse", "make_client"]
