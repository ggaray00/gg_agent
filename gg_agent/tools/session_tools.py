"""session_search — the model's recall over past conversations.

One tool, four modes, picked by which arguments are present:

  query                          → DISCOVER: best-matching past sessions
  session_id + around_message_id → SCROLL:   ±window messages around an anchor
  session_id                     → READ:     head + tail of one session
  (nothing)                      → BROWSE:   recent sessions

Everything is scoped to the agent's user: other users' sessions can't be found,
read or scrolled, and an id belonging to someone else reads as "no such session".

Search is PULLED by the model, never pushed: nothing is injected into the prompt
per turn, one line of system-prompt guidance says the tool exists. Every result
is real stored messages — no LLM summarization in between.

Mirrors hermes-agent: tools/session_search_tool.py
"""

from __future__ import annotations

from typing import Any

from ..persistence import SessionInfo, StoredMessage
from .registry import registry

DISCOVER_SCAN = 100            # hits fetched before per-session dedup
DEFAULT_SESSIONS, MAX_SESSIONS = 3, 10
DEFAULT_WINDOW, MAX_WINDOW = 5, 20
TOP_SESSION_WINDOW = 5
READ_HEAD, READ_TAIL = 20, 10
MAX_MESSAGE_CHARS = 4_000
BROWSE_DEFAULT = 10

SYNTAX_HELP = ('Search syntax: plain words match all of them; "quoted phrase" matches the phrase; '
               "`a OR b` matches either; `-term` excludes a term. Try fewer or different words.")


def _when(dt) -> str | None:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else None


def _clamp(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _render(m: StoredMessage) -> dict[str, Any]:
    content = m.content or ""
    if len(content) > MAX_MESSAGE_CHARS:
        content = content[:MAX_MESSAGE_CHARS] + f"… [truncated, {len(m.content)} chars]"
    out: dict[str, Any] = {"id": m.id, "role": m.role, "content": content}
    if m.tool_name:
        out["tool_name"] = m.tool_name
    if m.tool_calls:
        out["tool_calls"] = [tc.get("function", {}).get("name") for tc in m.tool_calls]
    return out


def _session_meta(info: SessionInfo) -> dict[str, Any]:
    return {"session_id": info.id, "title": info.title, "when": _when(info.started_at),
            "last_activity": _when(info.last_activity_at), "messages": info.message_count,
            "model": info.model, "cwd": info.cwd}


def _roles(role_filter: Any) -> tuple[str, ...]:
    if isinstance(role_filter, str):
        role_filter = role_filter.split(",")
    roles = tuple(r.strip() for r in (role_filter or ()) if r and r.strip() in {"user", "assistant", "tool"})
    return roles or ("user", "assistant")


async def _discover(store, agent, query: str, limit: int, roles) -> dict[str, Any]:
    hits = await store.search(query, limit=DISCOVER_SCAN, roles=roles, owner_id=agent.user_id,
                              exclude_session_ids=[agent.session_id], include_subagents=False)
    best: dict[str, Any] = {}
    for hit in hits:                          # already ranked: first hit per session is its best
        if hit.session_id not in best:
            best[hit.session_id] = hit
            if len(best) >= limit:
                break
    if not best:
        return {"mode": "discover", "query": query, "results": [],
                "message": f"No past conversations matched {query!r}. {SYNTAX_HELP}"}

    results = []
    for i, hit in enumerate(best.values()):
        entry = {"session_id": hit.session_id, "title": hit.title, "when": _when(hit.started_at),
                 "match_message_id": hit.message_id, "match_role": hit.role, "snippet": hit.snippet}
        if i == 0:
            # Adaptive detail: only the best session is hydrated with context.
            around = await store.messages_around(hit.session_id, hit.message_id, TOP_SESSION_WINDOW)
            entry["context"] = [_render(m) for m in around["messages"]]
        results.append(entry)
    return {"mode": "discover", "query": query, "results": results,
            "hint": ("To read more of a session, call session_search again with its session_id and "
                     "around_message_id=<match_message_id> (optionally window=N).")}


async def _owned(store, agent, session_id: str) -> SessionInfo | None:
    info = await store.get_session(session_id)
    return info if info is not None and info.owner_id == agent.user_id else None


async def _scroll(store, agent, session_id: str, anchor: int, window: int) -> dict[str, Any]:
    around = ({"messages": []} if await _owned(store, agent, session_id) is None
              else await store.messages_around(session_id, anchor, window))
    if not around["messages"]:
        return {"mode": "scroll", "session_id": session_id, "messages": [],
                "message": f"No messages found in session {session_id!r}."}
    return {"mode": "scroll", "session_id": session_id, "anchor_id": anchor,
            "messages": [_render(m) for m in around["messages"]],
            "messages_before": around["messages_before"], "messages_after": around["messages_after"],
            "hint": (f"messages_before/messages_after below {window} means the start/end of the "
                     "session was reached. Scroll further by anchoring on the first or last id.")}


async def _read(store, agent, session_id: str) -> dict[str, Any]:
    info = await _owned(store, agent, session_id)
    if info is None:
        return {"error": f"No session {session_id!r}. Call session_search with no arguments to browse."}
    rows = await store.get_messages(session_id)
    out: dict[str, Any] = {"mode": "read", **_session_meta(info)}
    if len(rows) <= READ_HEAD + READ_TAIL:
        out["messages"] = [_render(m) for m in rows]
    else:
        out["messages"] = [_render(m) for m in rows[:READ_HEAD]]
        out["omitted"] = len(rows) - READ_HEAD - READ_TAIL
        out["tail"] = [_render(m) for m in rows[-READ_TAIL:]]
        out["hint"] = "Middle omitted; use around_message_id to scroll into it."
    return out


async def _browse(store, agent, limit: int) -> dict[str, Any]:
    sessions = await store.list_sessions(limit=limit, owner_id=agent.user_id)
    return {"mode": "browse",
            "sessions": [{**_session_meta(s), "current": s.id == agent.session_id} for s in sessions],
            "hint": "Pass a session_id to read one, or a query to search across all of them."}


async def session_search(query: str | None = None, session_id: str | None = None,
                         around_message_id: int | None = None, window: int | None = None,
                         limit: int | None = None, role_filter: Any = None,
                         parent_agent: Any = None) -> dict[str, Any]:
    store = getattr(parent_agent, "store", None)
    if store is None:
        return {"error": "session_search is unavailable: session persistence is not enabled."}
    await parent_agent.astart()

    query = (query or "").strip()
    session_id = (session_id or "").strip()
    if session_id and around_message_id is not None:
        anchor = _clamp(around_message_id, 0, 0, 2**62)
        return await _scroll(store, parent_agent, session_id, anchor,
                             _clamp(window, DEFAULT_WINDOW, 1, MAX_WINDOW))
    if session_id:
        return await _read(store, parent_agent, session_id)
    if query:
        return await _discover(store, parent_agent, query,
                               _clamp(limit, DEFAULT_SESSIONS, 1, MAX_SESSIONS), _roles(role_filter))
    return await _browse(store, parent_agent, _clamp(limit, BROWSE_DEFAULT, 1, 50))


async def _handler(**kw) -> dict[str, Any]:
    return await session_search(
        query=kw.get("query"), session_id=kw.get("session_id"),
        around_message_id=kw.get("around_message_id"), window=kw.get("window"),
        limit=kw.get("limit"), role_filter=kw.get("role_filter"),
        parent_agent=kw.get("parent_agent"))


registry.register_toolset("sessions", "Recall past conversations from the session store")
registry.register(
    name="session_search",
    toolset="sessions",
    emoji="🔎",
    # Availability is per agent (does IT have a store?), so Agent.tool_definitions
    # hides the tool when persistence is off rather than a global check_fn.
    needs_agent=True,
    handler=_handler,
    schema={
        "name": "session_search",
        "description": (
            "Search and read PAST conversation history with the user (earlier sessions, not this "
            "one). Modes, chosen by the arguments you pass:\n"
            "- query → find the best-matching past sessions (the top one comes with context)\n"
            "- session_id + around_message_id → scroll a window of messages around that id\n"
            "- session_id alone → read the start and end of that session\n"
            "- no arguments → list recent sessions\n\n"
            "It only knows what was said in past conversations. Never use it to conclude that a "
            "file, function or fact doesn't exist — check the files or live sources first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": 'Keywords. Supports "quoted phrases", OR, and -excluded terms.'},
                "session_id": {"type": "string", "description": "A session id from an earlier result."},
                "around_message_id": {"type": "integer",
                                      "description": "Anchor message id (e.g. match_message_id) to scroll around."},
                "window": {"type": "integer",
                           "description": f"Messages each side of the anchor (default {DEFAULT_WINDOW}, max {MAX_WINDOW})."},
                "limit": {"type": "integer",
                          "description": f"Sessions to return (discover: default {DEFAULT_SESSIONS}, max {MAX_SESSIONS})."},
                "role_filter": {"type": "array", "items": {"type": "string", "enum": ["user", "assistant", "tool"]},
                                "description": "Roles to search (default user + assistant; add tool for tool output)."},
            },
            "required": [],
        },
    },
)
