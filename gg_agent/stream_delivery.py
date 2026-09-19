"""Stream delivery: what the user sees while a response is still arriving.

The transports push raw deltas; this module decides what reaches the event
stream, as these events:

    stream_delta     {text}    visible answer text, in order
    reasoning_delta  {text}    chain-of-thought (provider field or <think> tags)
    tool_gen_start   {name}    the model has started writing a call to this tool
    stream_break     {}        a segment ended; tools are about to run

Two jobs beyond forwarding:
  * ``ThinkScrubber`` pulls ``<think>…</think>`` spans out of the text, since
    several open models (DeepSeek-R1, Qwen, local Ollama builds) inline their
    reasoning in the content.
  * Segment breaks: after a tool round, the next text is prefixed with a blank
    line, so a consumer that just concatenates deltas gets readable paragraphs.

Only the display is scrubbed. History keeps the transport's own ``content``,
exactly as the non-streaming path does.

Mirrors hermes-agent: agent/stream_delivery.py + agent/think_scrubber.py
"""

from __future__ import annotations

from collections.abc import Callable

from .transports.streaming import StreamHooks

_TAGS = {"<think>": "</think>", "<thinking>": "</thinking>"}


def _held_suffix(buf: str, candidates: list[str]) -> int:
    """Length of the longest tail of ``buf`` that could still grow into one of
    ``candidates`` — the part to hold back until the next chunk decides it."""
    for size in range(min(len(buf), max(map(len, candidates)) - 1), 0, -1):
        tail = buf[-size:]
        if any(c.startswith(tail) for c in candidates):
            return size
    return 0


class ThinkScrubber:
    """Split streamed text into ``(visible, reasoning)``, tags removed.

    An open tag only counts at the start of a line, so prose that *mentions*
    ``<think>`` mid-sentence is left alone. A tag split across chunks is held
    back until it is complete, so nothing half-tagged is ever shown.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._close: str | None = None      # the closing tag while inside a block
        self._line_start = True             # is the char before _buf a line start?

    def feed(self, text: str) -> tuple[str, str]:
        self._buf += text
        visible: list[str] = []
        reasoning: list[str] = []
        while self._buf:
            if self._close is not None:
                end = self._buf.find(self._close)
                if end >= 0:
                    reasoning.append(self._buf[:end])
                    self._buf = self._buf[end + len(self._close):]
                    self._close, self._line_start = None, True
                    continue
                hold = _held_suffix(self._buf, [self._close])
                reasoning.append(self._buf[:len(self._buf) - hold])
                self._buf = self._buf[len(self._buf) - hold:]
                break

            start, tag = self._find_open()
            if tag is not None:
                self._emit_visible(visible, self._buf[:start])
                self._buf = self._buf[start + len(tag):]
                self._close = _TAGS[tag]
                continue
            hold = self._open_hold()
            self._emit_visible(visible, self._buf[:len(self._buf) - hold])
            self._buf = self._buf[len(self._buf) - hold:]
            break
        return "".join(visible), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        """End of stream: release whatever was held back."""
        buf, inside = self._buf, self._close is not None
        self._buf, self._close, self._line_start = "", None, True
        return ("", buf) if inside else (buf, "")

    def _at_line_start(self, i: int) -> bool:
        return self._line_start if i == 0 else self._buf[i - 1] == "\n"

    def _find_open(self) -> tuple[int, str | None]:
        best: tuple[int, str | None] = (-1, None)
        for tag in _TAGS:
            i = self._buf.find(tag)
            while i >= 0 and not self._at_line_start(i):
                i = self._buf.find(tag, i + 1)
            if i >= 0 and (best[1] is None or i < best[0]):
                best = (i, tag)
        return best

    def _open_hold(self) -> int:
        hold = _held_suffix(self._buf, list(_TAGS))
        return hold if hold and self._at_line_start(len(self._buf) - hold) else 0

    def _emit_visible(self, out: list[str], text: str) -> None:
        if text:
            out.append(text)
            self._line_start = text.endswith("\n")


class StreamDelivery:
    """Per-turn delivery state; ``begin_attempt`` resets the per-call part."""

    def __init__(self, emit: Callable[..., None]) -> None:
        self._emit = emit
        self._turn_streamed = False       # anything shown yet this turn?
        self._needs_break = False
        self.begin_attempt()

    def begin_attempt(self) -> None:
        self._parts: list[str] = []
        self._scrubber = ThinkScrubber()
        self.tool_started = False

    @property
    def delivered(self) -> bool:
        """Did this attempt show the user any answer text?"""
        return bool(self._parts)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def hooks(self, is_interrupted: Callable[[], bool]) -> StreamHooks:
        return StreamHooks(on_text=self.fire_text, on_reasoning=self.fire_reasoning,
                           on_tool_start=self.fire_tool_start, is_interrupted=is_interrupted)

    def fire_text(self, text: str) -> None:
        visible, reasoning = self._scrubber.feed(text)
        if reasoning:
            self.fire_reasoning(reasoning)
        self._deliver(visible)

    def fire_reasoning(self, text: str) -> None:
        if text:
            self._emit("reasoning_delta", text=text)

    def fire_tool_start(self, name: str) -> None:
        self.tool_started = True
        self._emit("tool_gen_start", name=name)

    def finish(self) -> None:
        """The call ended (normally or not): release held-back text."""
        visible, reasoning = self._scrubber.flush()
        self.fire_reasoning(reasoning)
        self._deliver(visible)

    def segment_break(self) -> None:
        """Tools are about to run. Close the visible segment, if one is open."""
        if self._turn_streamed:
            self._emit("stream_break")
            self._needs_break = True

    def _deliver(self, text: str) -> None:
        if not text:
            return
        if not self._parts:
            # A reply often opens with the newlines that followed a </think>.
            text = text.lstrip("\n")
            if not text:
                return
            if self._needs_break:
                text, self._needs_break = "\n\n" + text, False
        self._parts.append(text)
        self._turn_streamed = True
        self._emit("stream_delta", text=text)


__all__ = ["StreamDelivery", "ThinkScrubber"]
