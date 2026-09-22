"""The Agent: provider resolution, client lifecycle, tool grant, turn facade.

Deliberately kept as ONE class with plain attributes. hermes-agent's ``AIAgent``
is the same object assembled from ~15 mixins (client lifecycle, streaming,
interrupts, compression, persistence, ...) — when a concern here grows past a
screen, that's the seam to split on.

The API is async first — ``arun`` / ``aask`` / ``aclose``, and ``async with``.
The sync ``run`` / ``ask`` / ``close`` are thin wrappers over a background event
loop (see ``gg_agent.aio``) so scripts and tests need no ceremony. Calling a sync
one from inside a running loop raises rather than deadlocking.

Persistence is opt-in and lives behind ``self.store`` (see ``gg_agent.persistence``).
With no store the agent behaves exactly as if persistence did not exist.

Mirrors hermes-agent: run_agent.AIAgent + agent/agent_init.py + agent/client_lifecycle.py
+ agent/session_persistence.py
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from typing import Any

from .aio import run_sync
from .home import get_working_dir
from .loop import mark_persisted, run_conversation
from .persistence import SUBAGENT_SOURCE, SessionStore, get_default_store
from .prompts import build_system_prompt
from .providers import ProviderProfile, get_provider_profile, iter_configured
from .reasoning_effort import parse_reasoning
from .tools.registry import discover_builtin_tools, registry
from .transports import get_transport

logger = logging.getLogger(__name__)

# A leaf subagent must not delegate further, or a bad prompt can fork-bomb the
# machine. Depth, not the model, decides who gets this tool back.
DELEGATE_BLOCKED_TOOLS = {"delegate_task"}
# Subagents work from the context they were handed, not from past sessions.
SUBAGENT_BLOCKED_TOOLS = {"session_search"}


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
        # Only providers keyed by an env var are auto-picked: keyless local servers,
        # AWS/GCP/ChatGPT logins and CLI subprocesses must be asked for by name.
        if profile.key_env_vars:
            return profile
    raise RuntimeError(
        "No provider credentials found. Either:\n"
        "  • export OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY / ... "
        "(`gg-agent --list-providers` shows all of them)\n"
        "  • use GitHub Copilot: `gg-agent login`, or sign in via VS Code / the Copilot CLI\n"
        "  • name one that needs no key: -p ollama | bedrock | vertex | openai-codex | copilot-acp | opencode-free"
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
        # Reasoning: None = $GG_REASONING, else the provider's own default. A level
        # ("low" / "medium" / "high" / "xhigh" / "max"), "on", "off", or a ready-made
        # ``{"enabled": bool, "effort": str}``. Each profile translates it to its
        # provider's wire shape (see ``gg_agent.reasoning_effort``).
        reasoning: str | dict | None = None,
        # Prompt caching: mark the stable part of the prompt so the provider can
        # skip re-reading it. Only the Anthropic transport asks for it explicitly
        # (see transports/anthropic.py); every OpenAI-compatible endpoint caches
        # prefixes on its own either way. Off costs ~25% more on the first call
        # of a turn and saves ~90% on the rest, so it is worth turning off only
        # when a session really is one call long.
        prompt_caching: bool = True,
        # Context compression: None = $GG_COMPRESS (on unless "0"/"false"/"off").
        # Keeps a long session inside the model's window by pruning old tool
        # output before each request (see ``gg_agent.compression``).
        # ``context_length`` overrides the window this model is assumed to have;
        # without it the model name decides, then $GG_CONTEXT_LENGTH.
        compress: bool | None = None,
        context_length: int | None = None,
        cwd: str | None = None,
        event_callback: Callable[[str, dict], None] | None = None,
        # Streaming: None = $GG_STREAM (on unless "0"/"false"/"off"). Deltas go
        # to ``event_callback`` as stream_delta / reasoning_delta / tool_gen_start /
        # stream_break events. Subagents never stream: parallel children would
        # interleave their tokens on the same callback.
        stream: bool | None = None,
        # MCP: False = off, True = load .mcp.json, a path = load that file,
        # or a list of MCPServerConfig. Connection happens on the first turn,
        # because __init__ is sync and connecting is not.
        mcp: bool | str | os.PathLike | list | None = None,
        # Persistence: None = $GG_DATABASE_URL if set (else off), False = off,
        # or a SessionStore. Every session has an owner, so it also needs the
        # ``user_id`` of a registered user (see persistence/users.py) — without
        # one nothing is saved. ``resume`` loads one of that user's sessions on
        # the first turn (or call ``astart()`` to load it now).
        store: SessionStore | bool | None = None,
        user_id: str | None = None,
        resume: str | None = None,
        source: str = "cli",
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
            raise ValueError(f"No model given and provider {self.profile.name!r} has no default "
                             f"(pass one with -m; `gg-agent -p {self.profile.name} --list-models` lists them).")

        # Credentials come from the profile hook, not from an env var directly:
        # a provider whose token is minted per session (Copilot) resolves here.
        resolved_key, resolved_base_url = self.profile.resolve_credentials()
        self.api_key = api_key or resolved_key
        self.base_url = base_url or resolved_base_url or self.profile.base_url
        # Explicit values pin the client; only auto-resolved ones get refreshed.
        self._pinned_credentials = bool(api_key and base_url)
        self.max_tokens = max_tokens or self.profile.get_max_tokens(self.model)
        self.temperature = temperature
        if reasoning is None:
            reasoning = os.getenv("GG_REASONING") or None
        self.reasoning_config = reasoning if isinstance(reasoning, dict) else parse_reasoning(reasoning)
        self.prompt_caching = prompt_caching
        if compress is None:
            compress = os.getenv("GG_COMPRESS", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.compress = bool(compress)
        self.context_length = context_length
        # Estimated-to-billed prompt-token ratio, corrected after every response
        # that reports usage. 1.0 until the first one lands.
        self._token_scale = 1.0
        # Compression state: when the summarizer may be tried again, and how many
        # passes in a row reclaimed too little to be worth the cache miss.
        self._summary_cooldown_until = 0.0
        self._compression_strikes = 0

        self.session_id = uuid.uuid4().hex[:12]
        self.cwd = os.path.abspath(cwd) if cwd else get_working_dir()

        self.registry = registry
        self.enabled_toolsets = enabled_toolsets
        self.blocked_tools = set(blocked_tools or ())

        # A child shares its parent's store (and so its pool); only whoever
        # created the store from the environment closes it.
        self._owns_store = False
        if store is False:
            store = None
        elif store is None or store is True:
            if parent is not None:
                store = parent.store
            else:
                store = get_default_store()
                self._owns_store = store is not None
        # Subagents act for the same user as their parent.
        self.user_id = user_id if user_id is not None else (parent.user_id if parent else None)
        self._persistence_off_reason: str | None = None
        if store is not None and self.user_id is None:
            store, self._owns_store = None, False
            self._persistence_off_reason = "no user_id given, so sessions have no owner and are not saved"
        self.store: SessionStore | None = store
        self.source = SUBAGENT_SOURCE if parent is not None and source == "cli" else source
        self._pending_resume = resume
        self._store_ready = False
        self._session_created = False
        self._sessions_to_end: list[str] = []

        self.system_prompt = system_prompt or build_system_prompt(
            extra_instructions, cwd=self.cwd, session_search=self._session_search_enabled())

        self.max_iterations = max_iterations
        self.event_callback = event_callback
        self.depth, self.max_depth, self.parent = depth, max_depth, parent
        if stream is None:
            stream = os.getenv("GG_STREAM", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.streaming = bool(stream) and depth == 0
        # Set by the loop when the endpoint rejects streaming; sticks for the session.
        self._stream_disabled = False

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

    # ── Persistence ──────────────────────────────────────────────────────

    async def astart(self) -> bool:
        """Open the store (and load ``resume``) now instead of on the first turn.

        Returns whether persistence is active. An unreachable database disables
        persistence with a ``persist_error`` event rather than raising — except
        when resuming, where the transcript is the whole point: that raises.
        """
        if self.store is None:
            reason, self._persistence_off_reason = self._persistence_off_reason, None
            if self._pending_resume:
                raise RuntimeError(f"cannot resume {self._pending_resume}: " + (
                    reason or "persistence is not enabled (set GG_DATABASE_URL or pass store=)"))
            if reason:                                   # reported once
                self._persist_error("owner", RuntimeError(reason))
            return False
        if not self._store_ready:
            self._store_ready = True
            try:
                await self.store.open()
                # Checked up front (root agents only — children share the parent's
                # user): otherwise every session insert would fail on the foreign key.
                if self.parent is None and await self.store.get_user(self.user_id) is None:
                    raise LookupError(f"user {self.user_id!r} does not exist in this database")
            except Exception as exc:
                if self._pending_resume:
                    raise RuntimeError(f"cannot resume {self._pending_resume}: {exc}") from exc
                self._persist_error("owner" if isinstance(exc, LookupError) else "open", exc)
                if self._owns_store:
                    try:
                        await self.store.close()
                    except Exception:
                        logger.debug("store close failed", exc_info=True)
                self.store = None
                return False
            if self._pending_resume:
                session_id, self._pending_resume = self._pending_resume, None
                await self._load_session(session_id)
        return self.store is not None

    async def aresume(self, session_id: str) -> None:
        """Switch this agent to an earlier session: its id, its transcript."""
        if not await self.astart():
            raise RuntimeError("persistence is not enabled (set GG_DATABASE_URL or pass store=)")
        await self._flush_ended_sessions()
        await self._load_session(session_id)

    def resume(self, session_id: str) -> None:
        run_sync(self.aresume(session_id))

    async def _load_session(self, session_id: str) -> None:
        info = await self.store.get_session(session_id)
        # Someone else's session looks exactly like a missing one: no leaking ids.
        if info is None or info.owner_id != self.user_id:
            raise ValueError(f"No such session {session_id!r}")
        if self._session_created and self.session_id != session_id:
            self._sessions_to_end.append(self.session_id)
        self.session_id = session_id
        self.history = await self.store.load_history(session_id)
        # Everything that came out of the store is durable; marking it stops the
        # next flush from writing the whole transcript back a second time.
        mark_persisted(self.history)
        self._session_created = True

    async def _ensure_session(self, *, raise_errors: bool = False) -> None:
        """Create this session's row, once. Retried on every flush until it lands."""
        if self._session_created or self.store is None:
            return
        try:
            await self.store.create_session(
                self.session_id, owner_id=self.user_id, source=self.source,
                provider=self.profile.name, model=self.model,
                system_prompt=self.system_prompt, cwd=self.cwd,
                parent_session_id=self.parent.session_id if self.parent else None)
            self._session_created = True
        except Exception as exc:
            if raise_errors:
                raise
            self._persist_error("create", exc)

    async def _persist(self, messages: list[dict[str, Any]]) -> None:
        await self._ensure_session(raise_errors=True)
        await self.store.append_messages(self.session_id, messages)

    async def _flush_ended_sessions(self) -> None:
        """End sessions left behind by the sync ``reset()``, which can't await."""
        while self._sessions_to_end and self.store is not None:
            session_id = self._sessions_to_end.pop(0)
            try:
                await self.store.end_session(session_id, "new_session")
            except Exception as exc:
                self._persist_error("end", exc)

    def _persist_error(self, stage: str, exc: BaseException, **extra: Any) -> None:
        """Report a persistence failure once: as an event when someone listens
        (the CLI prints those), as a log warning when nobody does."""
        if self.event_callback is None:
            logger.warning("persistence %s failed for session %s: %s", stage, self.session_id, exc)
        else:
            logger.debug("persistence %s failed for session %s", stage, self.session_id, exc_info=exc)
        self._emit("persist_error", stage=stage, error=str(exc), **extra)

    # ── Tool grant ───────────────────────────────────────────────────────

    def tool_definitions(self) -> list[dict]:
        """Schemas this agent is allowed to use. The subagent restriction is
        exactly this call with a different filter — nothing deeper."""
        blocked = self.blocked_tools if self.store is not None else self.blocked_tools | {"session_search"}
        return self.registry.get_definitions(
            enabled_toolsets=self.enabled_toolsets,
            blocked_tools=blocked,
        )

    def _session_search_enabled(self) -> bool:
        return (self.store is not None and "session_search" not in self.blocked_tools
                and (self.enabled_toolsets is None or "sessions" in self.enabled_toolsets))

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
        # A stateless one-shot is not part of the session, so it isn't recorded.
        persisting = keep_history and await self.astart()
        if persisting:
            await self._flush_ended_sessions()
            await self._ensure_session()
        await self.refresh_credentials()
        result = await run_conversation(
            self, user_message, conversation_history=self.history,
            persist=self._persist if persisting else None)
        if keep_history:
            self.history = result["history"]
        if persisting and self._session_created:
            usage = result["usage"]
            try:
                await self.store.add_usage(self.session_id, usage.prompt_tokens, usage.completion_tokens)
            except Exception as exc:
                self._persist_error("usage", exc)
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
        """Clear the conversation. With a store, that means a NEW session; the old
        one is ended on the next turn or at close (this method is sync)."""
        self.history = []
        if self.store is not None:
            if self._session_created:
                self._sessions_to_end.append(self.session_id)
            self.session_id = uuid.uuid4().hex[:12]
            self._session_created = False

    # ── Teardown ─────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        for child in list(self._children):
            await child.aclose()
        try:
            await self.transport.aclose_client(self.client)
        except Exception:
            logger.debug("client close failed", exc_info=True)
        if self.store is not None and self._store_ready:
            await self._flush_ended_sessions()
            if self._session_created:
                try:
                    await self.store.end_session(self.session_id, "closed")
                except Exception as exc:
                    self._persist_error("end", exc)
            if self._owns_store:
                try:
                    await self.store.close()
                except Exception:
                    logger.debug("store close failed", exc_info=True)
                self._store_ready = False

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
