"""Central tool registry.

Each tool module calls ``registry.register()`` at import time to declare its
schema, handler and toolset membership. The loop queries the registry instead
of keeping parallel data structures, which is what makes toolsets (and the
subagent's restricted tool grant) a one-line filter.

Handlers may be ``def`` or ``async def``. ``dispatch`` is a coroutine either way:
an async handler is awaited, a sync one is pushed to a worker thread so a slow
tool (a subprocess, a big read) cannot stall the event loop that the rest of the
turn — and every other subagent — is sharing.

Import chain is cycle-safe: this module imports nothing from the agent;
tool modules import it; the agent imports both.

Mirrors hermes-agent: tools/registry.py + model_tools.handle_function_call
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import pkgutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Cap on a tool error body so a runaway exception can't eat the context window.
_MAX_TOOL_ERROR_CHARS = 2048
# Cap on a tool result: the model pays for every character of it, every turn after.
DEFAULT_MAX_RESULT_CHARS = 30_000


def tool_error(message: str) -> str:
    """Uniform error result. Errors go back to the MODEL, not up as exceptions —
    the model can usually recover (fix the path, retry with different args)."""
    if len(message) > _MAX_TOOL_ERROR_CHARS:
        message = message[:_MAX_TOOL_ERROR_CHARS] + "… [truncated]"
    return json.dumps({"error": message}, ensure_ascii=False)


def tool_ok(payload: Any) -> str:
    """Uniform success result as a JSON string."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, default=str)


def _is_async_handler(handler: Callable) -> bool:
    """True when the handler should be awaited rather than sent to a thread.

    ``registry.register(handler=lambda **kw: some_async_fn(**kw))`` is a common
    shape, and a lambda is never a coroutine function — so also treat a wrapper
    whose ``__wrapped__`` or single closure cell is async as async.
    """
    if inspect.iscoroutinefunction(handler):
        return True
    wrapped = getattr(handler, "__wrapped__", None)
    if wrapped is not None and inspect.iscoroutinefunction(wrapped):
        return True
    return bool(getattr(handler, "_gg_async", False))


@dataclass
class ToolEntry:
    name: str
    toolset: str
    schema: dict
    handler: Callable[..., Any]
    description: str = ""
    emoji: str = "🔧"
    check_fn: Callable[[], bool] | None = None
    needs_agent: bool = False              # handler receives parent_agent=<Agent>
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    is_async: bool = False                 # awaited directly; sync handlers go to a thread

    def available(self) -> bool:
        """False when a runtime requirement is missing (binary absent, key unset)."""
        try:
            return bool(self.check_fn()) if self.check_fn else True
        except Exception:
            return False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self._toolsets: dict[str, str] = {}    # toolset -> human description
        self._lock = threading.RLock()

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, name: str, toolset: str, schema: dict, handler: Callable,
                 *, check_fn: Callable | None = None, emoji: str = "🔧",
                 needs_agent: bool = False, description: str = "",
                 max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
                 override: bool = False) -> None:
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset and not override:
                logger.error(
                    "Tool registration REJECTED: %r (toolset %r) would shadow toolset %r",
                    name, toolset, existing.toolset)
                return
            self._tools[name] = ToolEntry(
                name=name, toolset=toolset, schema=schema, handler=handler,
                description=description or schema.get("description", ""),
                emoji=emoji, check_fn=check_fn, needs_agent=needs_agent,
                max_result_chars=max_result_chars,
                # Detected once, here — not on every call. A lambda wrapping a
                # coroutine function reads as sync, so unwrap that common case.
                is_async=_is_async_handler(handler),
            )

    def register_toolset(self, name: str, description: str) -> None:
        self._toolsets[name] = description

    # ── Queries ──────────────────────────────────────────────────────────

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def all_names(self) -> list[str]:
        return sorted(self._tools)

    def toolsets(self) -> dict[str, str]:
        return dict(self._toolsets)

    def get_definitions(self, enabled_toolsets: list[str] | None = None,
                        blocked_tools: set[str] | None = None) -> list[dict]:
        """Schemas the model is allowed to see this turn.

        This single filter is the whole permission model: a subagent gets a
        narrower ``enabled_toolsets`` / wider ``blocked_tools`` and physically
        cannot call what it wasn't granted.
        """
        blocked = blocked_tools or set()
        out = []
        for entry in sorted(self._tools.values(), key=lambda e: e.name):
            if entry.name in blocked:
                continue
            if enabled_toolsets is not None and entry.toolset not in enabled_toolsets:
                continue
            if not entry.available():
                continue
            out.append(entry.schema)
        return out

    # ── Dispatch ─────────────────────────────────────────────────────────

    async def dispatch(self, name: str, args: dict, *, agent: Any = None) -> str:
        """Execute one tool call; ALWAYS returns a string for the tool message.

        A raised exception here would kill the turn, so every failure is caught
        and handed back to the model as an error result instead. Cancellation is
        the one thing re-raised: an interrupt must not be swallowed as a tool error.
        """
        entry = self._tools.get(name)
        if entry is None:
            return tool_error(f"Unknown tool {name!r}. Available: {', '.join(self.all_names())}")
        if not entry.available():
            return tool_error(f"Tool {name!r} is currently unavailable (requirement check failed).")
        try:
            kwargs = dict(args or {})
            if entry.needs_agent:
                kwargs["parent_agent"] = agent
            if entry.is_async:
                result = await entry.handler(**kwargs)
            else:
                # to_thread keeps a blocking tool off the loop. Sibling tool calls,
                # the API request and other subagents all keep running meanwhile.
                result = await asyncio.to_thread(lambda: entry.handler(**kwargs))
                # A sync handler is still allowed to return an awaitable.
                if inspect.isawaitable(result):
                    result = await result
        except asyncio.CancelledError:
            raise
        except TypeError as exc:
            return tool_error(f"Bad arguments for {name}: {exc}")
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return tool_error(f"Error executing {name}: {exc}")

        text = tool_ok(result)
        if len(text) > entry.max_result_chars:
            text = (text[: entry.max_result_chars]
                    + f"\n… [truncated: {len(text)} chars total]")
        return text


registry = ToolRegistry()


def discover_builtin_tools() -> list[str]:
    """Import every sibling module in ``gg_agent.tools`` so it self-registers."""
    import gg_agent.tools as pkg

    imported = []
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name in {"registry"}:
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
            imported.append(mod.name)
        except Exception as exc:
            logger.warning("Could not import tool module %s: %s", mod.name, exc)
    return imported
