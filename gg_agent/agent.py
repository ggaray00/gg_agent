"""The Agent: provider resolution, client lifecycle, tool grant, turn facade.

Deliberately kept as ONE class with plain attributes. hermes-agent's ``AIAgent``
is the same object assembled from ~15 mixins (client lifecycle, streaming,
interrupts, compression, persistence, ...) — when a concern here grows past a
screen, that's the seam to split on.

The API is async first — ``arun`` / ``aask`` / ``aclose``, and ``async with``.
The sync ``run`` / ``ask`` / ``close`` are thin wrappers over a background event
loop (see ``gg_agent.aio``) so scripts and tests need no ceremony. Calling a sync
one from inside a running loop raises rather than deadlocking.

Mirrors hermes-agent: run_agent.AIAgent + agent/agent_init.py + agent/client_lifecycle.py
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from typing import Any

from .aio import run_sync
from .loop import run_conversation
from .prompts import build_system_prompt
from .providers import ProviderProfile, get_provider_profile, iter_configured
from .tools.registry import discover_builtin_tools, registry
from .transports import get_transport

logger = logging.getLogger(__name__)

# A leaf subagent must not delegate further, or a bad prompt can fork-bomb the
# machine. Depth, not the model, decides who gets this tool back.
DELEGATE_BLOCKED_TOOLS = {"delegate_task"}


def resolve_provider(name: str | None = None) -> ProviderProfile:
    """Pick a provider: explicit name, then $GG_PROVIDER, then the first one whose
    credentials are actually present in the environment."""
    requested = name or os.getenv("GG_PROVIDER", "")
    if requested:
        profile = get_provider_profile(requested)
        if profile is None:
            raise ValueError(f"Unknown provider {requested!r}")
        return profile
    for profile in iter_configured():
        if profile.env_vars:          # skip keyless local providers in auto-detect
            return profile
    raise RuntimeError(
        "No provider credentials found. Either:\n"
        "  • export OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY\n"
        "  • use GitHub Copilot: `gg-agent login`, or sign in via VS Code / the Copilot CLI\n"
        "  • run a local model: provider='ollama'"
    )


class Agent:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        extra_instructions: str = "",
        enabled_toolsets: list[str] | None = None,
        blocked_tools: set[str] | None = None,
        max_iterations: int = 50,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cwd: str | None = None,
        event_callback: Callable[[str, dict], None] | None = None,
        # MCP: False = off, True = load .mcp.json, a path = load that file,
        # or a list of MCPServerConfig. Connection happens on the first turn,
        # because __init__ is sync and connecting is not.
        mcp: bool | str | os.PathLike | list | None = None,
        # Delegation lineage — set by the delegate tool, not by users.
        depth: int = 0,
        max_depth: int = 2,
        parent: Agent | None = None,
    ) -> None:
        self.profile = resolve_provider(provider)
        self.api_mode = self.profile.api_mode
        self.transport = get_transport(self.api_mode)

        self.model = model or self.profile.default_model
        if not self.model:
            raise ValueError(f"No model given and provider {self.profile.name!r} has no default.")

        # Credentials come from the profile hook, not from an env var directly:
        # a provider whose token is minted per session (Copilot) resolves here.
        resolved_key, resolved_base_url = self.profile.resolve_credentials()
        self.api_key = api_key or resolved_key
        self.base_url = base_url or resolved_base_url or self.profile.base_url
        # Explicit values pin the client; only auto-resolved ones get refreshed.
        self._pinned_credentials = bool(api_key and base_url)
        self.max_tokens = max_tokens or self.profile.default_max_tokens
        self.temperature = temperature

        self.session_id = uuid.uuid4().hex[:12]
        self.cwd = os.path.abspath(cwd) if cwd else os.getcwd()
        self.system_prompt = system_prompt or build_system_prompt(extra_instructions, cwd=self.cwd)

        self.registry = registry
        self.enabled_toolsets = enabled_toolsets
        self.blocked_tools = set(blocked_tools or ())

        self.max_iterations = max_iterations
        self.event_callback = event_callback
        self.depth, self.max_depth, self.parent = depth, max_depth, parent

        self.history: list[dict[str, Any]] = []
        self._interrupt = threading.Event()
        self._children: list[Agent] = []

        # Subagents inherit the parent's setting but never reconnect: the pool is
        # process-wide, so by the time a child runs, the servers are already up.
        self.mcp = mcp
        self._mcp_connected = bool(parent is not None and mcp)

        discover_builtin_tools()
        self.client = self._build_client()
        logger.info("Agent %s ready: %s / %s (%s)",
                    self.session_id, self.profile.name, self.model, self.api_mode)

    # ── Credentials ──────────────────────────────────────────────────────

    def _build_client(self) -> Any:
        return self.transport.build_client(
            api_key=self.api_key, base_url=self.base_url, profile=self.profile,
        )

    async def refresh_credentials(self) -> bool:
        """Re-resolve credentials and rebuild the client if they changed.

        Called before every turn. Copilot's exchanged token expires in ~30 minutes,
        so a long session would otherwise start 401-ing mid-conversation; the profile
        hook hands back a freshly exchanged token and the client is swapped under us.
        Returns True when the client was replaced.
        """
        if self._pinned_credentials:
            return False
        try:
            api_key, base_url = self.profile.resolve_credentials()
        except Exception as exc:
            # A refresh failure is not fatal — the existing client may still be valid.
            logger.debug("credential refresh failed, keeping current client: %s", exc)
            return False
        base_url = base_url or self.profile.base_url
        if not api_key or (api_key == self.api_key and base_url == self.base_url):
            return False
        old_client = self.client
        self.api_key, self.base_url = api_key, base_url
        self.client = self._build_client()
        logger.info("credentials refreshed for %s; client rebuilt", self.profile.name)
        try:
            await self.transport.aclose_client(old_client)
        except Exception:
            logger.debug("old client close failed", exc_info=True)
        return True

    # ── MCP ──────────────────────────────────────────────────────────────

    async def connect_mcp(self) -> dict[str, str]:
        """Connect the configured MCP servers and register their tools.

        Idempotent and safe to call directly; ``arun`` calls it on the first turn.
        Returns ``{server: status}``. A server that fails to start is reported in
        that mapping, not raised — one bad entry must not stop the agent.
        """
        from .tools.mcp_tools import MCPServerConfig, connect_mcp_servers

        self._mcp_connected = True
        if isinstance(self.mcp, list):
            configs = [c if isinstance(c, MCPServerConfig) else MCPServerConfig.from_dict(**c)
                       for c in self.mcp]
            status = await connect_mcp_servers(configs=configs)
        else:
            path = None if self.mcp is True else self.mcp
            status = await connect_mcp_servers(config_path=path)
        for name, state in status.items():
            self._emit("mcp_server", server=name, status=state)
        return status

    # ── Tool grant ───────────────────────────────────────────────────────

    def tool_definitions(self) -> list[dict]:
        """Schemas this agent is allowed to use. The subagent restriction is
        exactly this call with a different filter — nothing deeper."""
        return self.registry.get_definitions(
            enabled_toolsets=self.enabled_toolsets,
            blocked_tools=self.blocked_tools,
        )

    def can_delegate(self) -> bool:
        return self.depth < self.max_depth and "delegate_task" not in self.blocked_tools

    # ── Interrupts ───────────────────────────────────────────────────────

    @property
    def interrupted(self) -> bool:
        return self._interrupt.is_set()

    def interrupt(self) -> None:
        """Cooperative stop; propagates to running children."""
        self._interrupt.set()
        for child in list(self._children):
            child.interrupt()

    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    # ── Events ───────────────────────────────────────────────────────────

    def _emit(self, kind: str, **payload: Any) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(kind, {"agent": self.session_id, "depth": self.depth, **payload})
        except Exception:
            logger.debug("event callback raised", exc_info=True)

    # ── Turn facade ──────────────────────────────────────────────────────

    async def arun(self, user_message: str, *, keep_history: bool = True) -> dict[str, Any]:
        """Run one turn. ``keep_history=False`` gives a stateless one-shot call."""
        if self.mcp and not self._mcp_connected:
            await self.connect_mcp()
        await self.refresh_credentials()
        result = await run_conversation(self, user_message, conversation_history=self.history)
        if keep_history:
            self.history = result["history"]
        return result

    async def aask(self, user_message: str, **kwargs) -> str:
        """Convenience: just the text of the final answer."""
        return (await self.arun(user_message, **kwargs))["response"] or ""

    def run(self, user_message: str, **kwargs) -> dict[str, Any]:
        """Synchronous ``arun``, for scripts and tests. Not usable inside a loop."""
        return run_sync(self.arun(user_message, **kwargs))

    def ask(self, user_message: str, **kwargs) -> str:
        return run_sync(self.aask(user_message, **kwargs))

    def reset(self) -> None:
        self.history = []

    # ── Teardown ─────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        for child in list(self._children):
            await child.aclose()
        try:
            await self.transport.aclose_client(self.client)
        except Exception:
            logger.debug("client close failed", exc_info=True)

    def close(self) -> None:
        run_sync(self.aclose())

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<Agent {self.session_id} {self.profile.name}/{self.model} "
                f"depth={self.depth}>")
