"""Transport selection by api_mode.

Adding a protocol = one ``ProviderTransport`` subclass registered here. The
loop never changes.

Mirrors hermes-agent: agent/transports/__init__.py
"""

from __future__ import annotations

from .base import ProviderTransport
from .types import NormalizedResponse, ToolCall, Usage, build_tool_call, map_finish_reason

_TRANSPORTS: dict[str, ProviderTransport] = {}


def register_transport(transport: ProviderTransport) -> ProviderTransport:
    _TRANSPORTS[transport.api_mode] = transport
    return transport


def get_transport(api_mode: str) -> ProviderTransport:
    transport = _TRANSPORTS.get(api_mode or "chat_completions")
    if transport is None:
        raise ValueError(
            f"No transport for api_mode={api_mode!r}. Known: {sorted(_TRANSPORTS)}"
        )
    return transport


def _register_builtins() -> None:
    from .chat_completions import ChatCompletionsTransport
    register_transport(ChatCompletionsTransport())
    try:
        from .anthropic import AnthropicTransport
    except Exception:  # anthropic SDK not installed — chat_completions still works
        return
    register_transport(AnthropicTransport())


_register_builtins()

__all__ = [
    "ProviderTransport", "NormalizedResponse", "ToolCall", "Usage",
    "build_tool_call", "map_finish_reason", "get_transport", "register_transport",
]
