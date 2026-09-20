"""Shared types for normalized provider responses.

The loop only ever sees these — never a raw SDK object. Only fields every
consumer reads are top-level; protocol-specific state lives in ``provider_data``.

Mirrors hermes-agent: agent/transports/types.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A normalized tool call from any provider.

    ``id`` is the protocol's canonical identifier (``tool_call_id`` /
    ``tool_use_id``). ``arguments`` is always a JSON *string*, as in the
    OpenAI wire format, so the loop has one shape to parse.
    """

    id: str | None
    name: str
    arguments: str
    provider_data: dict[str, Any] | None = field(default=None, repr=False)

    # Back-compat accessors: code written against the OpenAI SDK reads
    # ``tc.function.name`` / ``tc.function.arguments``.
    type = property(lambda self: "function")
    function = property(lambda self: self)


def _field(obj: Any, name: str) -> int:
    """Read ``name`` off an SDK model or a plain dict; missing/None -> 0."""
    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return value or 0


@dataclass
class Usage:
    """Token counts, normalized across providers.

    ``prompt_tokens`` is ALWAYS the whole input — cached and uncached alike —
    so totals stay comparable. The providers disagree on this: OpenAI counts
    cache hits inside ``prompt_tokens``, Anthropic reports the uncached
    remainder only and leaves you to add the cache fields back. Both are folded
    into the OpenAI convention here, and the split is kept below.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0        # of prompt_tokens, served from cache (~0.1x price)
    cache_write_tokens: int = 0   # of prompt_tokens, written to cache (~1.25x price)

    @classmethod
    def from_openai(cls, u: Any) -> Usage:
        """Chat Completions usage, including whatever cache hits it reports.

        Every OpenAI-compatible endpoint caches prefixes automatically, but they
        report it in two different places: OpenAI/Groq/OpenRouter nest a
        ``prompt_tokens_details.cached_tokens``, DeepSeek puts
        ``prompt_cache_hit_tokens`` at the top level. Neither is a separate
        charge to add — both are a discount on tokens already in
        ``prompt_tokens``.
        """
        if u is None:
            return cls()
        details = u.get("prompt_tokens_details") if isinstance(u, dict) else getattr(u, "prompt_tokens_details", None)
        cached = _field(details, "cached_tokens") if details is not None else 0
        return cls(
            prompt_tokens=_field(u, "prompt_tokens"),
            completion_tokens=_field(u, "completion_tokens"),
            total_tokens=_field(u, "total_tokens"),
            cached_tokens=cached or _field(u, "prompt_cache_hit_tokens"),
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class NormalizedResponse:
    """Normalized API response from any provider — the transport layer's only return type."""

    content: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: str            # "stop" | "tool_calls" | "length" | "content_filter"
    reasoning: str | None = None
    usage: Usage | None = None
    provider_data: dict[str, Any] | None = field(default=None, repr=False)


def build_tool_call(id: str | None, name: str, arguments: Any, **provider_fields: Any) -> ToolCall:
    """Build a ``ToolCall``; dict arguments are JSON-serialised."""
    args_str = json.dumps(arguments) if isinstance(arguments, (dict, list)) else str(arguments or "{}")
    return ToolCall(id=id, name=name, arguments=args_str,
                    provider_data=dict(provider_fields) if provider_fields else None)


def map_finish_reason(reason: str | None, mapping: dict[str, str]) -> str:
    """Translate a provider stop reason; unknown or None -> "stop"."""
    return "stop" if reason is None else mapping.get(reason, "stop")
