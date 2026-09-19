"""The agentic loop — one user turn, run to completion.

This is the heart of the whole thing, and it is genuinely small:

    while budget remains:
        assemble request  ->  call model  ->  normalize response
        if the model asked for tools:  run them, append results, loop again
        else:                          that text is the answer, stop

Everything else in a production agent (compression, failover, checkpoints,
approval gates, streaming) hangs off those four phases. hermes-agent splits each
phase into its own ``agent/turn_*.py`` module and threads a ``_LoopState``
dataclass through them; the same shape is kept here at one-file scale so the
seams are visible.

Mirrors hermes-agent: agent/conversation_loop.py + agent/turn_*.py + agent/tool_executor.py
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .stream_delivery import StreamDelivery
from .transports.streaming import StreamInterrupted, is_stream_unsupported
from .transports.types import NormalizedResponse, Usage

logger = logging.getLogger(__name__)

MAX_TOOL_WORKERS = 8          # concurrent tool executions per batch
MAX_API_RETRIES = 4


@dataclass
class LoopState:
    """Every local the turn threads through the phases.

    A phase reads the fields it needs and rebinds the ones it owns. Keeping them
    in one object (rather than as 30 locals) is what lets each phase move into
    its own module without changing a signature.
    """

    messages: list[dict[str, Any]]
    api_call_count: int = 0
    final_response: str | None = None
    finish_reason: str = "stop"
    interrupted: bool = False
    failed: bool = False
    exit_reason: str = "unknown"
    usage: Usage = field(default_factory=Usage)
    tool_calls_made: int = 0
    # Per-iteration slots, rebound by the phases before any later phase reads them.
    api_kwargs: dict[str, Any] | None = None
    response: NormalizedResponse | None = None
    # Persistence: a callback that makes messages durable, and the index into
    # ``messages`` of the first one it hasn't accepted yet.
    persist: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None
    persisted_upto: int = 0
    # Streaming: what the user has been shown, and whether the final answer was
    # among it (so a caller that rendered the deltas doesn't print it twice).
    stream: StreamDelivery | None = None
    streamed: bool = False


# ── Phase 1: assemble the request ────────────────────────────────────────────

def assemble_request(agent, s: LoopState) -> None:
    """Build the provider-native kwargs for this iteration.

    Note what is NOT here: message history is the loop's own OpenAI-shaped list.
    Only the transport knows what the wire format looks like.
    """
    s.api_kwargs = agent.transport.build_kwargs(
        model=agent.model,
        messages=s.messages,
        tools=agent.tool_definitions(),
        profile=agent.profile,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
    )


# ── Phase 2: call the model (with its own retry loop) ────────────────────────

async def perform_api_call(agent, s: LoopState) -> NormalizedResponse | None:
    """One model call with bounded retries. Returns None when every attempt failed.

    Retries are the transport's problem only in the sense of *what* to re-send;
    *whether* to retry is a loop decision, which is why the SDK clients are all
    built with ``max_retries=0``.

    Streaming changes the retry rules, because text on screen can't be taken back:
      * an endpoint that rejects streaming switches it off and retries — free;
      * a failure AFTER text was shown is not retried (it would repeat the text):
        the partial answer is kept as the response, ``finish_reason="length"``;
        the exception is a drop mid tool-call, which is retried with a notice;
      * an interrupt mid-stream keeps the partial answer in history.
    """
    last_error: Exception | None = None
    attempt = 0
    while attempt < MAX_API_RETRIES:
        if agent.interrupted:
            s.interrupted, s.exit_reason = True, "interrupted"
            return None
        streaming = _should_stream(agent, s)
        if s.stream is not None:
            s.stream.begin_attempt()
        try:
            agent._emit("api_call", iteration=s.api_call_count, model=agent.model, stream=streaming)
            if streaming:
                hooks = s.stream.hooks(lambda: agent.interrupted)
                response = await agent.transport.call_stream(agent.client, hooks, **s.api_kwargs)
                s.stream.finish()
                return response
            raw = agent.transport.call(agent.client, **s.api_kwargs)
            # Awaited only when it is awaitable: a scripted test transport can
            # stay a plain function without needing async def.
            if inspect.isawaitable(raw):
                raw = await raw
            return agent.transport.normalize_response(raw)
        except asyncio.CancelledError:
            s.interrupted = True
            raise
        except StreamInterrupted:
            s.stream.finish()
            s.interrupted, s.exit_reason = True, "interrupted"
            if s.stream.delivered:
                partial = s.stream.text
                record_assistant_message(s, NormalizedResponse(
                    content=partial, tool_calls=None, finish_reason="interrupted"))
                s.final_response, s.streamed = partial, True
            return None
        except Exception as exc:
            last_error = exc
            if streaming:
                s.stream.finish()
                if not s.stream.delivered and is_stream_unsupported(exc):
                    logger.warning("streaming rejected by %s, falling back: %s", agent.profile.name, exc)
                    agent._stream_disabled = True
                    agent._emit("api_retry", error=f"streaming unsupported ({exc}); retrying without it",
                                delay=0.0)
                    continue                     # not counted: nothing was wrong with the request
                if s.stream.delivered and not (s.stream.tool_started and _is_retryable(exc)):
                    agent._emit("stream_error", error=str(exc))
                    return NormalizedResponse(content=s.stream.text, tool_calls=None, finish_reason="length")
                if s.stream.delivered:
                    agent._emit("stream_reset", error=str(exc))
            attempt += 1
            if not _is_retryable(exc) or attempt >= MAX_API_RETRIES:
                break
            delay = min(2 ** (attempt - 1), 16) + random.random()
            logger.warning("API call failed (%s), retrying in %.1fs", exc, delay)
            agent._emit("api_retry", error=str(exc), delay=delay)
            await asyncio.sleep(delay)

    s.failed = True
    s.exit_reason = "api_error"
    s.final_response = f"[API error after {attempt} attempts: {last_error}]"
    return None


def _should_stream(agent, s: LoopState) -> bool:
    return (s.stream is not None and getattr(agent, "streaming", False)
            and agent.transport.supports_streaming and not getattr(agent, "_stream_disabled", False))


def _is_retryable(exc: Exception) -> bool:
    """Rate limits, overloads and transient network faults are worth another try;
    a 400 on a malformed request is not — retrying it just burns the budget."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 408 or status == 429 or status >= 500
    text = str(exc).lower()
    return any(k in text for k in
               ("rate limit", "overloaded", "timeout", "timed out", "connection", "temporarily"))


# ── Phase 3: fold the response into history ──────────────────────────────────

def record_assistant_message(s: LoopState, response: NormalizedResponse) -> dict[str, Any]:
    """Append the assistant turn in OpenAI shape — the loop's canonical format."""
    message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
    if response.tool_calls:
        message["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in response.tool_calls
        ]
    s.messages.append(message)
    return message


# ── Phase 4: run the tools ───────────────────────────────────────────────────

async def run_tool_round(agent, s: LoopState, response: NormalizedResponse) -> None:
    """Execute every tool the model asked for, append one tool message each.

    Two invariants worth keeping when you extend this:
      1. The assistant message is appended BEFORE any tool runs, so a crash
         mid-batch leaves a resumable history rather than an orphaned result.
      2. EVERY tool call gets exactly one ``role:"tool"`` reply, even on failure.
         A missing reply makes the next request invalid on every provider.
    """
    calls = response.tool_calls or []
    if s.stream is not None:
        s.stream.segment_break()
    agent._emit("tool_round", count=len(calls))

    # A semaphore, not a thread pool: the cap is on how many tools run at once,
    # and the registry decides per tool whether that means awaiting a coroutine
    # or occupying a worker thread.
    gate = asyncio.Semaphore(MAX_TOOL_WORKERS)

    async def execute(tc) -> str:
        try:
            args = json.loads(tc.arguments or "{}")
            if not isinstance(args, dict):
                args = {}
        except (TypeError, ValueError) as exc:
            return f'{{"error": "invalid JSON arguments: {exc}"}}'
        async with gate:
            agent._emit("tool_start", name=tc.name, args=args)
            started = time.monotonic()
            result = await agent.registry.dispatch(tc.name, args, agent=agent)
            agent._emit("tool_end", name=tc.name, result=result,
                        duration=time.monotonic() - started)
            return result

    # gather preserves input order, which matters: the API requires tool replies
    # in the same order as the tool_calls they answer.
    results = await asyncio.gather(*(execute(tc) for tc in calls))

    for tc, result in zip(calls, results, strict=True):
        s.messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": result if result is not None else '{"error": "tool produced no result"}',
        })
    s.tool_calls_made += len(calls)


# ── Persistence ──────────────────────────────────────────────────────────────

async def flush(agent, s: LoopState) -> None:
    """Hand the not-yet-durable tail of the transcript to ``s.persist``.

    A failure is logged and reported, never raised: losing the transcript is bad,
    losing the turn as well is worse. ``persisted_upto`` only advances on success,
    so the same tail is offered again at the next flush point.

    Index-based dedup is sound only because history is append-only. Compression
    or rewind would need hermes's per-message marker (agent/session_persistence.py).
    """
    end = len(s.messages)
    if s.persist is None or s.persisted_upto >= end:
        return
    pending = [m for m in s.messages[s.persisted_upto:end] if m.get("role") != "system"]
    try:
        if pending:
            await s.persist(pending)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        agent._persist_error("append", exc, pending=len(pending))
        return
    s.persisted_upto = end


# ── The loop itself ──────────────────────────────────────────────────────────

async def run_conversation(agent, user_message: str,
                           conversation_history: list[dict[str, Any]] | None = None,
                           system_prompt: str | None = None,
                           persist: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None,
                           persisted_history: int | None = None) -> dict[str, Any]:
    """Run one user turn to completion. Returns the result dict the caller keeps.

    ``persist`` makes messages durable (the loop knows nothing about stores).
    ``persisted_history`` says how many of ``conversation_history`` already are —
    all of them unless an earlier turn's flush failed.
    """
    history_in = conversation_history or []
    messages: list[dict[str, Any]] = []
    system = system_prompt or agent.system_prompt
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history_in)
    messages.append({"role": "user", "content": user_message})

    offset = 1 if system else 0
    durable = len(history_in) if persisted_history is None else min(persisted_history, len(history_in))
    s = LoopState(messages=messages, persist=persist, persisted_upto=offset + durable,
                  stream=StreamDelivery(agent._emit))
    started = time.time()

    while s.api_call_count < agent.max_iterations:
        if agent.interrupted:
            s.interrupted, s.exit_reason = True, "interrupted"
            break

        assemble_request(agent, s)
        s.api_call_count += 1

        response = await perform_api_call(agent, s)
        if response is None:
            break
        s.response, s.finish_reason = response, response.finish_reason
        if response.usage:
            s.usage = s.usage + response.usage

        record_assistant_message(s, response)
        # Durable BEFORE any tool runs: a crash mid-round leaves a transcript that
        # resume can repair, instead of tool side effects with no record of the ask.
        await flush(agent, s)

        if response.tool_calls:
            await run_tool_round(agent, s, response)
            await flush(agent, s)
            continue                        # back to the model with the results

        # No tool calls: this text is the answer.
        s.final_response = response.content or ""
        s.streamed = s.stream is not None and s.stream.delivered
        s.exit_reason = "final_response"
        agent._emit("final", text=s.final_response)
        break
    else:
        s.exit_reason = "max_iterations"
        s.final_response = (s.final_response
                            or f"[stopped after {agent.max_iterations} iterations without a final answer]")

    # ── Finalize ─────────────────────────────────────────────────────────
    # Catches the exits that skip the flush points (interrupt, API error, iteration
    # cap) and retries anything an earlier flush failed to write.
    await flush(agent, s)

    # The history handed back EXCLUDES the system message: the next turn rebuilds
    # it, so a prompt change takes effect immediately instead of being pinned by
    # a stale copy in the transcript.
    history = [m for m in s.messages if m.get("role") != "system"]
    return {
        "response": s.final_response,
        "history": history,
        "api_calls": s.api_call_count,
        "tool_calls": s.tool_calls_made,
        "usage": s.usage,
        "duration_seconds": round(time.time() - started, 2),
        "interrupted": s.interrupted,
        "failed": s.failed,
        # True when ``response`` already reached the event stream as stream_delta events.
        "streamed": s.streamed,
        "exit_reason": s.exit_reason,
        # How many entries of ``history`` are durable (== len(history) unless a flush failed).
        "persisted": max(0, s.persisted_upto - offset),
    }
