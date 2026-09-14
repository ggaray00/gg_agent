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


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    @classmethod
    def from_openai(cls, u: Any) -> Usage:
        if u is None:
            return cls()
        return cls(**{k: getattr(u, k, 0) or 0
                      for k in ("prompt_tokens", "completion_tokens", "total_tokens")})

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
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
