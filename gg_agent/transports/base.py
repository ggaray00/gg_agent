"""Abstract base for provider transports.

A transport owns ONE api_mode's data path:

    convert_messages -> convert_tools -> build_kwargs -> normalize_response

The client is an ASYNC SDK client and ``call`` is a coroutine; the loop awaits
it. A transport whose ``call`` is a plain function still works — the loop awaits
the result only when it is awaitable — which keeps scripted test doubles simple.

Streaming is optional: a transport that sets ``supports_streaming`` also
implements ``call_stream``, which pushes deltas through ``StreamHooks`` while it
builds the response and then returns the same ``NormalizedResponse`` that
``normalize_response`` would have. The loop picks the path; nothing downstream
can tell which one ran.

It does NOT own credentials, retries or interrupts — those stay on Agent. That
split is what lets the agentic loop stay provider-agnostic: it always speaks
OpenAI-shaped messages and always receives a ``NormalizedResponse`` back.

Mirrors hermes-agent: agent/transports/base.py
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

from .types import NormalizedResponse


class ProviderTransport(ABC):
    # True when ``call_stream`` is implemented.
    supports_streaming: bool = False

    # Provider stop_reason -> OpenAI finish_reason. None = already OpenAI vocabulary.
    _STOP_REASON_MAP: dict[str, str] | None = None

    @property
    @abstractmethod
    def api_mode(self) -> str:
        """The api_mode string this transport handles."""

    @abstractmethod
    def build_client(self, *, api_key: str, base_url: str, profile: Any) -> Any:
        """Construct the async SDK client for this protocol."""

    @abstractmethod
    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        """OpenAI-format messages -> provider-native structure."""

    @abstractmethod
    def convert_tools(self, tools: list[dict[str, Any]]) -> Any:
        """OpenAI-format tool definitions -> provider-native format."""

    @abstractmethod
    def build_kwargs(self, model: str, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None, **params) -> dict[str, Any]:
        """Primary entry point: kwargs ready to hand to the SDK call."""

    @abstractmethod
    async def call(self, client: Any, **api_kwargs) -> Any:
        """Perform the request. Separate from build_kwargs so retries can re-send."""

    async def call_stream(self, client: Any, hooks: Any, **api_kwargs) -> NormalizedResponse:
        """Perform the request streaming: feed ``hooks`` (a ``StreamHooks``) as
        deltas arrive, return the assembled response. Raise ``StreamInterrupted``
        when ``hooks.is_interrupted()`` turns true mid-stream."""
        raise NotImplementedError(f"{type(self).__name__} does not stream")

    async def aclose_client(self, client: Any) -> None:
        """Release the client's connection pool. Override if the SDK differs."""
        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    @abstractmethod
    def normalize_response(self, response: Any) -> NormalizedResponse:
        """Raw provider response -> NormalizedResponse."""

    def map_finish_reason(self, raw_reason: str | None) -> str:
        if self._STOP_REASON_MAP is None:
            return raw_reason or "stop"
        return self._STOP_REASON_MAP.get(raw_reason or "", "stop")
