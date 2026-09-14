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
from dataclasses import dataclass, field
from typing import Any

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
    """
    last_error: Exception | None = None
    for attempt in range(MAX_API_RETRIES):
        if agent.interrupted:
            s.interrupted = True
            return None
        try:
            agent._emit("api_call", iteration=s.api_call_count, model=agent.model)
            raw = agent.transport.call(agent.client, **s.api_kwargs)
            # Awaited only when it is awaitable: a scripted test transport can
            # stay a plain function without needing async def.
            if inspect.isawaitable(raw):
                raw = await raw
            return agent.transport.normalize_response(raw)
        except asyncio.CancelledError:
            s.interrupted = True
            raise
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt == MAX_API_RETRIES - 1:
                break
            delay = min(2 ** attempt, 16) + random.random()
            logger.warning("API call failed (%s), retrying in %.1fs", exc, delay)
            agent._emit("api_retry", error=str(exc), delay=delay)
            await asyncio.sleep(delay)

    s.failed = True
    s.exit_reason = "api_error"
    s.final_response = f"[API error after {MAX_API_RETRIES} attempts: {last_error}]"
    return None


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


# ── The loop itself ──────────────────────────────────────────────────────────

async def run_conversation(agent, user_message: str,
                           conversation_history: list[dict[str, Any]] | None = None,
                           system_prompt: str | None = None) -> dict[str, Any]:
    """Run one user turn to completion. Returns the result dict the caller keeps."""
    messages: list[dict[str, Any]] = []
    system = system_prompt or agent.system_prompt
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(conversation_history or [])
    messages.append({"role": "user", "content": user_message})

    s = LoopState(messages=messages)
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

        if response.tool_calls:
            await run_tool_round(agent, s, response)
            continue                        # back to the model with the results

        # No tool calls: this text is the answer.
        s.final_response = response.content or ""
        s.exit_reason = "final_response"
        agent._emit("final", text=s.final_response)
        break
    else:
        s.exit_reason = "max_iterations"
        s.final_response = (s.final_response
                            or f"[stopped after {agent.max_iterations} iterations without a final answer]")

    # ── Finalize ─────────────────────────────────────────────────────────
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
        "exit_reason": s.exit_reason,
    }
