"""Langfuse tracing — one trace per user turn, opt-in, never on the critical path.

What a turn looks like in Langfuse:

    gg-agent  (agent)                 input = user message, output = final answer
    ├─ openai/gpt-4.1  (generation)   messages in, text + tool calls out, token usage
    ├─ run_shell  (tool)              args in, result out
    ├─ delegate_task  (tool)
    │   └─ subagent  (agent)          children nest here: OTel context follows asyncio tasks
    │       └─ ...
    ├─ compression-summary  (generation)
    └─ openai/gpt-4.1  (generation)

Traces are grouped by ``session_id`` (the agent's session) and ``user_id``, so a
REPL conversation reads as one Langfuse session.

Enabled when the ``langfuse`` package is installed (``uv sync --extra tracing``)
and ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are set; ``GG_TRACING=0``
turns it off anyway. ``LANGFUSE_BASE_URL`` points it at a self-hosted server.
With tracing off every hook below is a no-op and ``langfuse`` is never imported.

A tracing failure is logged and swallowed: losing a trace must never cost a turn.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Transports whose provider reports cache reads/writes Anthropic-style; Langfuse
# prices those models by these usage keys. Everyone else follows OpenAI's naming.
_ANTHROPIC_USAGE_MODES = {"anthropic_messages", "bedrock_converse"}


class _NoopObservation:
    def update(self, **_: Any) -> _NoopObservation:
        return self


NOOP = _NoopObservation()


class NoopTracer:
    """Tracing off. Same surface as ``Tracer``, does nothing."""

    enabled = False

    @contextmanager
    def turn(self, agent: Any, user_message: str) -> Iterator[Any]:
        yield NOOP

    def end_turn(self, obs: Any, agent: Any, result: dict[str, Any]) -> None:
        pass

    @contextmanager
    def generation(self, agent: Any, *, name: str | None = None, model: str | None = None,
                   messages: list[dict[str, Any]] | None = None, **_: Any) -> Iterator[Any]:
        yield NOOP

    def end_generation(self, obs: Any, agent: Any, response: Any, *, error: str | None = None) -> None:
        pass

    @contextmanager
    def tool(self, name: str, args: dict[str, Any]) -> Iterator[Any]:
        yield NOOP

    def end_tool(self, obs: Any, result: str | None) -> None:
        pass

    def flush(self) -> None:
        pass


class Tracer(NoopTracer):
    """Tracing on, backed by a ``langfuse.Langfuse`` client (or anything shaped like one)."""

    enabled = True

    def __init__(self, client: Any) -> None:
        self.client = client

    # ── Turn ─────────────────────────────────────────────────────────────

    @contextmanager
    def turn(self, agent: Any, user_message: str) -> Iterator[Any]:
        """The root observation of a turn. A subagent's turn nests under the
        parent's ``delegate_task`` tool span and inherits its trace attributes."""
        is_root = agent.parent is None
        metadata = {"provider": agent.profile.name, "api_mode": agent.api_mode,
                    "depth": agent.depth, "session_id": agent.session_id}
        with ExitStack() as stack:
            try:
                obs = stack.enter_context(self.client.start_as_current_observation(
                    name="gg-agent" if is_root else "subagent", as_type="agent",
                    input=user_message, metadata=metadata))
                if is_root:
                    from langfuse import propagate_attributes

                    stack.enter_context(propagate_attributes(
                        session_id=agent.session_id,
                        user_id=agent.user_id,
                        trace_name="gg-agent",
                        tags=[agent.profile.name, agent.source],
                    ))
            except Exception:
                logger.debug("tracing: could not start turn", exc_info=True)
                obs = NOOP
            yield obs

    def end_turn(self, obs: Any, agent: Any, result: dict[str, Any]) -> None:
        usage = result.get("usage")
        _safe(obs.update,
              output=result.get("response"),
              level="ERROR" if result.get("failed") else None,
              status_message=result.get("exit_reason"),
              metadata={
                  "exit_reason": result.get("exit_reason"),
                  "api_calls": result.get("api_calls"),
                  "tool_calls": result.get("tool_calls"),
                  "interrupted": result.get("interrupted"),
                  "prompt_tokens": getattr(usage, "prompt_tokens", None),
                  "completion_tokens": getattr(usage, "completion_tokens", None),
              })

    # ── Model calls ──────────────────────────────────────────────────────

    @contextmanager
    def generation(self, agent: Any, *, name: str | None = None, model: str | None = None,
                   messages: list[dict[str, Any]] | None = None,
                   tools: list[dict] | None = None, **_: Any) -> Iterator[Any]:
        """One model call, retries included (they are one request from the loop's view)."""
        model = model or agent.model
        params = {k: v for k, v in {
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "reasoning": _reasoning_label(getattr(agent, "reasoning_config", None)),
            "stream": bool(getattr(agent, "streaming", False)),
        }.items() if v is not None}
        try:
            cm = self.client.start_as_current_observation(
                name=name or f"{agent.profile.name}/{model}", as_type="generation",
                model=model, input=_clean(messages),
                model_parameters=params,
                metadata={"tools": [t.get("name") or t.get("function", {}).get("name")
                                    for t in tools or []]} if tools else None)
            obs = cm.__enter__()
        except Exception:
            logger.debug("tracing: could not start generation", exc_info=True)
            yield NOOP
            return
        try:
            yield obs
        except BaseException as exc:
            _safe(obs.update, level="ERROR", status_message=str(exc) or type(exc).__name__)
            _safe_exit(cm, exc)
            raise
        _safe_exit(cm, None)

    def end_generation(self, obs: Any, agent: Any, response: Any, *, error: str | None = None) -> None:
        if response is None:
            _safe(obs.update, level="ERROR", status_message=error or "no response")
            return
        output: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            output["tool_calls"] = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                    for tc in response.tool_calls]
        if response.reasoning:
            output["reasoning"] = response.reasoning
        _safe(obs.update,
              output=output,
              usage_details=_usage_details(response.usage, agent.api_mode),
              level="WARNING" if response.finish_reason in ("length", "content_filter") else None,
              metadata={"finish_reason": response.finish_reason})

    # ── Tools ────────────────────────────────────────────────────────────

    @contextmanager
    def tool(self, name: str, args: dict[str, Any]) -> Iterator[Any]:
        try:
            cm = self.client.start_as_current_observation(name=name, as_type="tool", input=args)
            obs = cm.__enter__()
        except Exception:
            logger.debug("tracing: could not start tool span", exc_info=True)
            yield NOOP
            return
        try:
            yield obs
        except BaseException as exc:
            _safe(obs.update, level="ERROR", status_message=str(exc) or type(exc).__name__)
            _safe_exit(cm, exc)
            raise
        _safe_exit(cm, None)

    def end_tool(self, obs: Any, result: str | None) -> None:
        # Tools hand errors back to the model as ``{"error": ...}`` rather than raising.
        failed = isinstance(result, str) and result.lstrip().startswith('{"error"')
        _safe(obs.update, output=result, level="WARNING" if failed else None)

    def flush(self) -> None:
        _safe(self.client.flush)


# ── Construction ─────────────────────────────────────────────────────────────

_default: NoopTracer | None = None


def tracing_requested() -> bool:
    if os.getenv("GG_TRACING", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_default_tracer() -> NoopTracer:
    """The process-wide tracer, built from the environment on first use.

    One Langfuse client per process: it owns a background exporter thread, and
    every agent (children included) should feed the same one.
    """
    global _default
    if _default is None:
        _default = _build_from_env()
    return _default


def _build_from_env() -> NoopTracer:
    if not tracing_requested():
        return NoopTracer()
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("LANGFUSE_* keys are set but the langfuse package is not installed; "
                       "tracing is off (`uv sync --extra tracing`).")
        return NoopTracer()
    try:
        client = Langfuse(environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or None)
    except Exception as exc:
        logger.warning("Langfuse client failed to start; tracing is off: %s", exc)
        return NoopTracer()
    logger.info("Langfuse tracing on")
    return Tracer(client)


def resolve_tracer(tracing: Any, parent: Any = None) -> NoopTracer:
    """``Agent(tracing=...)``: None/True = the parent's tracer, else the
    environment's; False = off; or a ready-made tracer."""
    if tracing is False:
        return NoopTracer()
    if isinstance(tracing, NoopTracer):
        return tracing
    if parent is not None:
        return parent.tracer
    return get_default_tracer()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Drop loop-private ``_`` keys (persistence markers etc.) from what is traced."""
    if messages is None:
        return None
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def _usage_details(usage: Any, api_mode: str) -> dict[str, int] | None:
    """Langfuse sums cost per usage key, so ``input`` must be the UNcached part —
    ``Usage.prompt_tokens`` includes cache reads and writes (see transports/types.py)."""
    if usage is None:
        return None
    cached, written = usage.cached_tokens, usage.cache_write_tokens
    details = {
        "input": max(usage.prompt_tokens - cached - written, 0),
        "output": usage.completion_tokens,
    }
    if api_mode in _ANTHROPIC_USAGE_MODES:
        details["cache_read_input_tokens"] = cached
        details["cache_creation_input_tokens"] = written
    else:
        details["input_cached_tokens"] = cached
    return {k: v for k, v in details.items() if v}


def _reasoning_label(config: Any) -> str | None:
    if not isinstance(config, dict):
        return None
    if config.get("enabled") is False:
        return "off"
    return config.get("effort") or ("on" if config.get("enabled") else None)


def _safe(fn: Any, **kwargs: Any) -> None:
    try:
        fn(**{k: v for k, v in kwargs.items() if v is not None})
    except Exception:
        logger.debug("tracing call failed", exc_info=True)


def _safe_exit(cm: Any, exc: BaseException | None) -> None:
    try:
        if exc is None:
            cm.__exit__(None, None, None)
        else:
            cm.__exit__(type(exc), exc, exc.__traceback__)
    except Exception:
        logger.debug("tracing: span close failed", exc_info=True)
