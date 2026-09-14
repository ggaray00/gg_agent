"""MCP — expose Model Context Protocol servers as ordinary registry tools.

The whole integration is a translation layer. Once an MCP tool is registered it
is indistinguishable from ``read_file``: the loop dispatches it the same way,
toolset filtering governs it the same way, and a subagent's tool grant restricts
it the same way. Nothing in agent.py, loop.py or the transports knows MCP exists.

Three things are worth understanding here.

**Naming.** MCP tool names are only unique within their server, and two servers
will happily both export ``search``. Every tool is registered as
``<server>__<tool>`` and placed in the toolset ``mcp:<server>``, so servers can
never shadow each other and ``enabled_toolsets=["mcp:github"]`` is a real grant.

**Lifecycle.** A stdio server is a subprocess whose session is bound to the task
that opened it, and the anyio cancel scopes underneath require that the same task
close it. So each server gets one long-lived runner task that opens the streams,
signals ready, then parks on a shutdown event. Tool calls from other tasks just
send messages over the already-open session, which is safe. This is also why the
pool is process-wide: a fan-out of four subagents shares one set of servers
instead of forking four copies of every one.

**Failure.** A server that dies must not take the turn with it. Its tools stay
registered but report unavailable via ``check_fn``, which drops them from the
next request's tool list — the model simply stops being offered them.

Mirrors hermes-agent: plugins/mcp/* (which adds OAuth, resources, prompts and
per-server approval policies).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import registry, tool_error, tool_ok

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 30.0        # seconds to wait for initialize + tools/list
CALL_TIMEOUT = 120.0          # seconds for one tools/call
DEFAULT_CONFIG_FILES = (".mcp.json", ".claude/mcp.json")

# OpenAI requires ^[a-zA-Z0-9_-]{1,64}$ for function names; MCP is laxer.
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize(name: str) -> str:
    return _SAFE_NAME.sub("_", name)[:48]


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """First attribute that exists, by any of its spellings.

    The MCP SDK renamed its model fields from camelCase to snake_case
    (``inputSchema`` -> ``input_schema``). Reading only one spelling fails
    SILENTLY on the other version — an empty tool schema, so the model calls
    every tool with no arguments — so read both.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class MCPServerConfig:
    """One server, in the same shape other MCP clients use in their config files."""

    name: str
    command: str = ""                       # stdio: the executable
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""                           # stdio: working directory for the subprocess
    url: str = ""                           # http: streamable-HTTP endpoint instead
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @property
    def kind(self) -> str:
        return "http" if self.url else "stdio"

    @classmethod
    def from_dict(cls, name: str, blob: dict[str, Any]) -> MCPServerConfig:
        return cls(
            name=name,
            command=blob.get("command", "") or "",
            args=list(blob.get("args") or []),
            # ${VAR} in an env value is expanded from the real environment, so a
            # config file can be committed without the secret in it.
            env={k: os.path.expandvars(str(v)) for k, v in (blob.get("env") or {}).items()},
            cwd=os.path.expandvars(str(blob.get("cwd") or "")) or "",
            url=blob.get("url", "") or "",
            headers={k: os.path.expandvars(str(v)) for k, v in (blob.get("headers") or {}).items()},
            enabled=blob.get("enabled", True) is not False,
        )

    def problem(self) -> str | None:
        """Why this server can't be started, or None when it looks runnable."""
        if self.url:
            return None
        if not self.command:
            return "neither `command` (stdio) nor `url` (http) is set"
        if shutil.which(self.command) is None:
            return f"command not found on PATH: {self.command!r}"
        return None


def load_mcp_config(path: str | os.PathLike | None = None) -> list[MCPServerConfig]:
    """Read ``.mcp.json``. Same ``{"mcpServers": {...}}`` shape other clients use,
    so a config can be pasted across tools unchanged."""
    candidates = [Path(path)] if path else [Path(p) for p in DEFAULT_CONFIG_FILES]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            blob = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("could not read MCP config %s: %s", candidate, exc)
            return []
        servers = blob.get("mcpServers") or blob.get("servers") or {}
        return [MCPServerConfig.from_dict(name, cfg or {}) for name, cfg in servers.items()]
    return []


# ── One connected server ─────────────────────────────────────────────────────


class MCPServer:
    """A live connection, held open by its own runner task."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.session: Any = None
        self.tools: list[Any] = []
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def alive(self) -> bool:
        return self.session is not None and self._task is not None and not self._task.done()

    async def connect(self) -> bool:
        """Start the runner and wait until the session is initialized."""
        problem = self.config.problem()
        if problem:
            self.error = problem
            return False
        self._task = asyncio.create_task(self._serve(), name=f"mcp:{self.name}")
        ready = asyncio.create_task(self._ready.wait())
        # Wait for ready OR for the runner to die — whichever happens first, so a
        # server that exits instantly fails fast instead of burning the timeout.
        done, pending = await asyncio.wait(
            {ready, self._task}, timeout=CONNECT_TIMEOUT, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if task is ready:
                task.cancel()
        if not self._ready.is_set():
            self.error = self.error or f"timed out after {CONNECT_TIMEOUT:.0f}s"
            await self.aclose()
            return False
        return True

    async def _serve(self) -> None:
        """Open the transport and session, then hold them until shutdown.

        Everything is entered AND exited in this one task, which is what the
        anyio cancel scopes under the MCP SDK require.
        """
        try:
            from mcp import ClientSession
        except ImportError:
            self.error = "the `mcp` package is not installed (pip install mcp)"
            return

        try:
            async with self._transport() as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.tools = list((await session.list_tools()).tools)
                    logger.info("MCP %s: %d tool(s)", self.name, len(self.tools))
                    self._ready.set()
                    await self._shutdown.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("MCP %s failed: %s", self.name, self.error)
        finally:
            self.session = None
            self._ready.set()        # unblock connect() even on failure

    @contextlib.asynccontextmanager
    async def _transport(self):
        cfg = self.config
        if cfg.kind == "http":
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
            # Headers ride on the http client; the transport takes no headers kwarg.
            http_client = create_mcp_http_client(headers=cfg.headers or None)
            async with http_client:
                async with streamable_http_client(cfg.url, http_client=http_client) as streams:
                    yield streams[0], streams[1]
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
            params = StdioServerParameters(
                command=cfg.command, args=cfg.args,
                # Inherit the parent environment: servers routinely need PATH,
                # HOME and a proxy config to work at all.
                env={**os.environ, **cfg.env} if cfg.env else None,
                cwd=cfg.cwd or None,
            )
            async with stdio_client(params) as streams:
                yield streams[0], streams[1]

    async def call(self, tool_name: str, arguments: dict) -> str:
        if not self.alive:
            return tool_error(f"MCP server {self.name!r} is not connected ({self.error or 'stopped'})")
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments or {}), timeout=CALL_TIMEOUT)
        except asyncio.TimeoutError:
            return tool_error(f"{self.name}/{tool_name} timed out after {CALL_TIMEOUT:.0f}s")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return tool_error(f"{self.name}/{tool_name} failed: {type(exc).__name__}: {exc}")
        return _render_result(result)

    async def aclose(self) -> None:
        self._shutdown.set()
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
        except Exception:
            logger.debug("MCP %s close failed", self.name, exc_info=True)


def _render_result(result: Any) -> str:
    """MCP returns content blocks; the model wants one string.

    Text blocks are concatenated. Anything else (images, embedded resources) is
    described rather than inlined — a base64 image would blow the context window
    for no benefit, since the loop has nowhere to put it.
    """
    if _attr(result, "is_error", "isError", default=False):
        return tool_error(_text_of(result) or "the MCP server reported an error")
    structured = _attr(result, "structured_content", "structuredContent")
    if structured:
        return tool_ok(structured)
    return _text_of(result) or "(no content)"


def _text_of(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(getattr(block, "text", "") or "")
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None)
            parts.append(text if text else f"[resource: {getattr(resource, 'uri', 'unknown')}]")
        else:
            parts.append(f"[{kind or 'unknown'} content omitted]")
    return "\n".join(p for p in parts if p).strip()


# ── The pool ─────────────────────────────────────────────────────────────────


class MCPPool:
    """Process-wide set of connected servers.

    Process-wide on purpose: subagents share the parent's servers. One pool means
    one subprocess per server no matter how wide the fan-out gets.
    """

    def __init__(self) -> None:
        self.servers: dict[str, MCPServer] = {}
        self._lock = asyncio.Lock()

    def is_alive(self, name: str) -> bool:
        server = self.servers.get(name)
        return bool(server and server.alive)

    async def connect(self, configs: list[MCPServerConfig]) -> dict[str, str]:
        """Connect every enabled server in parallel and register its tools.

        Returns ``{server: status}`` — one line per server, connected or not.
        A failing server is reported, never raised: one broken entry in
        ``.mcp.json`` must not stop the agent from starting.
        """
        async with self._lock:
            wanted = [c for c in configs if c.enabled and c.name not in self.servers]
            if not wanted:
                return {name: self._status(name) for name in self.servers}

            servers = [MCPServer(c) for c in wanted]
            results = await asyncio.gather(
                *(s.connect() for s in servers), return_exceptions=True)

            for server, ok in zip(servers, results, strict=True):
                if isinstance(ok, BaseException):
                    server.error = f"{type(ok).__name__}: {ok}"
                    ok = False
                self.servers[server.name] = server
                if ok:
                    _register_server_tools(self, server)
            return {name: self._status(name) for name in self.servers}

    def _status(self, name: str) -> str:
        server = self.servers[name]
        if server.alive:
            return f"connected · {len(server.tools)} tool(s)"
        return f"unavailable · {server.error or 'not connected'}"

    async def aclose(self) -> None:
        servers, self.servers = list(self.servers.values()), {}
        await asyncio.gather(*(s.aclose() for s in servers), return_exceptions=True)


pool = MCPPool()


def _register_server_tools(owner: MCPPool, server: MCPServer) -> None:
    for tool in server.tools:
        raw_name = _attr(tool, "name", default="") or ""
        if not raw_name:
            continue
        qualified = f"{_sanitize(server.name)}__{_sanitize(raw_name)}"

        async def handler(_server=server, _tool=raw_name, **kwargs) -> str:
            kwargs.pop("parent_agent", None)
            return await _server.call(_tool, kwargs)

        registry.register(
            name=qualified,
            toolset=f"mcp:{server.name}",
            emoji="🔌",
            handler=handler,
            # Re-checked before every request: a server that died mid-session
            # drops out of the tool list instead of erroring on each call.
            check_fn=lambda _name=server.name: owner.is_alive(_name),
            description=_attr(tool, "description", default="") or "",
            schema={
                "name": qualified,
                "description": (_attr(tool, "description", default="")
                                or f"{raw_name} (via {server.name})"),
                "parameters": _attr(tool, "input_schema", "inputSchema",
                                    default=None) or {"type": "object", "properties": {}},
            },
            override=True,      # reconnecting replaces the previous registration
        )
    registry.register_toolset(f"mcp:{server.name}", f"MCP server {server.name!r}")


# ── Entry points ─────────────────────────────────────────────────────────────


async def connect_mcp_servers(config_path: str | os.PathLike | None = None,
                              configs: list[MCPServerConfig] | None = None) -> dict[str, str]:
    """Connect the configured servers and register their tools. Idempotent."""
    return await pool.connect(configs if configs is not None else load_mcp_config(config_path))


async def close_mcp_servers() -> None:
    await pool.aclose()


def _close_at_exit() -> None:
    """Reap stdio subprocesses on the way out.

    Registered AFTER ``gg_agent.aio`` imports, so atexit's LIFO order runs this
    while the background loop is still alive to run the coroutine on.
    """
    if not pool.servers:
        return
    from ..aio import in_async_context, run_sync
    if in_async_context():
        return                    # an async caller owns the lifecycle; leave it alone
    with contextlib.suppress(Exception):
        run_sync(close_mcp_servers(), timeout=10)


atexit.register(_close_at_exit)


def mcp_status() -> dict[str, str]:
    return {name: pool._status(name) for name in pool.servers}
