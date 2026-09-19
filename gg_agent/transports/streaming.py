"""Streaming plumbing shared by the transports.

A streaming transport still returns ONE ``NormalizedResponse`` — the loop cannot
tell a streamed answer from a blocking one. What streaming adds is a side channel:
while the response is being assembled, text, reasoning and "a tool call is being
written" are pushed through ``StreamHooks`` as they arrive.

The fiddly part is tool calls. Chat Completions sends them as fragments keyed by
``index``: the id and name usually arrive once, the JSON arguments in pieces that
only parse once joined. ``ToolCallAccumulator`` reassembles them.

Mirrors hermes-agent: agent/chat_completion_helpers.py (_ToolCallAccumulator,
_StreamingCall, _maybe_disable_streaming)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .types import ToolCall


def _noop(_: str) -> None:
    return None


@dataclass
class StreamHooks:
    """What a transport calls while a response streams in. All optional."""

    on_text: Callable[[str], None] = _noop
    on_reasoning: Callable[[str], None] = _noop
    on_tool_start: Callable[[str], None] = _noop     # a tool name, the first time it is known
    is_interrupted: Callable[[], bool] = field(default=lambda: False)


class StreamInterrupted(Exception):
    """The user interrupted mid-stream. Whatever was shown so far is the partial answer."""


class StreamDropped(ConnectionError):
    """The stream ended before the response was complete (e.g. mid tool-call).

    A ``ConnectionError`` whose message says "connection", so the loop's retry
    classifier treats it as transient."""


def is_stream_unsupported(exc: BaseException) -> bool:
    """True when the endpoint rejected streaming itself, not the request.

    Some OpenAI-compatible servers (and some proxies) refuse ``stream=True`` or
    ``stream_options``; the fix is to stop streaming, not to give up."""
    text = str(exc).lower()
    return ("stream" in text and ("not supported" in text or "unsupported" in text)) \
        or "stream_options" in text


class ToolCallAccumulator:
    """Reassemble Chat Completions tool-call deltas into complete ``ToolCall``s.

    Two provider quirks it absorbs:
      * Ollama reuses ``index`` 0 for parallel calls — a NEW id at an index that
        already has one means a new call, so it gets a fresh slot.
      * Some servers resend the full name in every delta, so the name is
        assigned, never appended.
    Argument fragments are collected in a list and joined once (``+=`` per chunk
    is quadratic on long arguments).
    """

    def __init__(self) -> None:
        self._slots: dict[int, dict[str, Any]] = {}
        self._slot_by_index: dict[int, int] = {}
        self._last_id_at_index: dict[int, str] = {}
        self._announced: set[int] = set()

    def __bool__(self) -> bool:
        return bool(self._slots)

    def feed(self, delta: Any) -> str | None:
        """Absorb one delta. Returns the tool name the first time it becomes known."""
        index = getattr(delta, "index", None) or 0
        call_id = getattr(delta, "id", None)
        call_id = str(call_id) if call_id is not None else None     # some servers send ints

        seen_id = self._last_id_at_index.get(index)
        if index not in self._slot_by_index or (call_id and seen_id and call_id != seen_id):
            self._slot_by_index[index] = len(self._slots)
        if call_id:
            self._last_id_at_index[index] = call_id

        slot = self._slot_by_index[index]
        entry = self._slots.setdefault(slot, {"id": None, "name": "", "args": []})
        if call_id:
            entry["id"] = call_id
        fn = getattr(delta, "function", None)
        name = getattr(fn, "name", None)
        if name:
            entry["name"] = name
        arguments = getattr(fn, "arguments", None)
        if arguments:
            entry["args"].append(arguments)

        if entry["name"] and slot not in self._announced:
            self._announced.add(slot)
            return entry["name"]
        return None

    def materialize(self) -> tuple[list[ToolCall], bool]:
        """``(tool_calls, truncated)``. Arguments that don't parse become ``"{}"``
        and set ``truncated`` — the stream was cut before the JSON closed."""
        calls: list[ToolCall] = []
        truncated = False
        for slot in sorted(self._slots):
            entry = self._slots[slot]
            if not entry["name"]:
                continue
            arguments = "".join(entry["args"]) or "{}"
            try:
                json.loads(arguments)
            except ValueError:
                arguments, truncated = "{}", True
            calls.append(ToolCall(id=entry["id"], name=entry["name"], arguments=arguments))
        return calls, truncated


async def aclose_quietly(stream: Any) -> None:
    """Close an SDK stream, releasing its connection. Never raises."""
    close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception:
        pass
