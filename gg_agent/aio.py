"""The bridge between the async core and synchronous callers.

The agent is async all the way down — the loop, the transports, the tools and
the MCP sessions all live on an event loop. Plenty of callers are not: a pytest
function, a `python scripts/01_hello.py`, the REPL. This module is the one place
that crosses between them.

Why a PERSISTENT background loop instead of ``asyncio.run()`` per call: an MCP
server is a long-lived subprocess whose session is bound to the loop that opened
it. ``asyncio.run()`` creates and destroys a loop per invocation, which would
tear down every MCP connection after each turn. One loop, running on a daemon
thread for the life of the process, keeps them alive between turns.

Async callers never touch any of this — they ``await agent.arun(...)`` and stay
on their own loop.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _LoopThread:
    """A lazily-started daemon thread running one event loop forever."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        return self._loop is not None

    def loop(self) -> asyncio.AbstractEventLoop:
        # Double-checked so the common path (already started) takes no lock.
        if self._loop is not None:
            return self._loop
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=self._run, args=(loop,), name="gg-agent-aio", daemon=True)
                thread.start()
                self._loop, self._thread = loop, thread
        return self._loop

    @staticmethod
    def _run(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop())  # type: ignore[return-value]

    def shutdown(self) -> None:
        """Stop the loop. Registered with atexit so MCP subprocesses get reaped."""
        loop, thread = self._loop, self._thread
        if loop is None:
            return
        self._loop = self._thread = None
        try:
            # Finalize async generators first (e.g. an HTTP stream body left
            # half-read), or their cleanup is scheduled on a loop that never runs it.
            asyncio.run_coroutine_threadsafe(loop.shutdown_asyncgens(), loop).result(timeout=2)
        except Exception:
            logger.debug("async generator shutdown failed", exc_info=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None:
                thread.join(timeout=5)
            loop.close()
        except Exception:
            logger.debug("loop shutdown failed", exc_info=True)


_BRIDGE = _LoopThread()
atexit.register(_BRIDGE.shutdown)


def in_async_context() -> bool:
    """True when the caller is already running inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def run_sync(coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
    """Run an async call from synchronous code and return its result.

    Raises if called from inside a running loop: silently nesting loops is the
    classic way to deadlock, and the caller there has a perfectly good ``await``.
    """
    if in_async_context():
        coro.close()
        raise RuntimeError(
            "run_sync() called from inside a running event loop. "
            "Use the async form instead — `await agent.arun(...)`, `await agent.aclose()`."
        )
    return _BRIDGE.submit(coro).result(timeout)


def shutdown() -> None:
    """Stop the background loop (also runs at interpreter exit)."""
    _BRIDGE.shutdown()


__all__ = ["run_sync", "in_async_context", "shutdown"]
