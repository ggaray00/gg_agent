"""Postgres ``SessionStore`` — psycopg 3, one async connection pool.

The pool is bound to the event loop that opens it, exactly like an MCP session,
so it is opened lazily on the agent's first turn (``Agent.__init__`` is sync) and
sync callers get it on ``gg_agent.aio``'s persistent background loop.

What Postgres gives for free that hermes's SQLite store builds by hand:
  * concurrency — a transaction-scoped advisory lock per session serializes
    writers to one transcript (hermes: compression locks + turn leases);
  * search — ``websearch_to_tsquery`` parses quotes / OR / -term and never raises
    on user input (hermes: ~40 lines of FTS5 query sanitizing), with a trigram
    index behind an ILIKE fallback for substrings the tokenizer can't see.

Mirrors hermes-agent: hermes_state_messages.py + hermes_state_search.py
"""

from __future__ import annotations

import logging
import re

from psycopg import AsyncConnection, errors, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from . import SUBAGENT_SOURCE, SearchHit, SessionInfo, StoredMessage, UserInfo, history_from_rows, make_title
from .migrate import migrate
from .serialize import count_tool_calls, message_to_row, snippet_around

logger = logging.getLogger(__name__)

OPEN_TIMEOUT_SECONDS = 5.0
_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_USER_COLUMNS = "id, email, password_hash, created_at"
_SESSION_COLUMNS = ("id, owner_id, source, parent_session_id, provider, model, system_prompt, cwd, title, "
                    "started_at, ended_at, end_reason, message_count, tool_call_count, "
                    "input_tokens, output_tokens, last_activity_at, metadata")
_MESSAGE_COLUMNS = "id, session_id, role, content, tool_calls, tool_call_id, tool_name, created_at"

_SEARCH_SQL = """
WITH q AS (SELECT websearch_to_tsquery('simple', %(q)s) AS q),
top AS (
  SELECT m.id, m.session_id, m.role, m.content, ts_rank_cd(m.search_tsv, q.q) AS rank
  FROM messages m
  JOIN sessions s ON s.id = m.session_id, q
  WHERE m.search_tsv @@ q.q
    AND m.active
    AND m.role = ANY(%(roles)s::text[])
    AND NOT (m.session_id = ANY(%(exclude)s::text[]))
    AND (%(include_subagents)s OR s.source <> %(subagent)s)
    AND (%(owner)s::text IS NULL OR s.owner_id = %(owner)s::text)
  ORDER BY rank DESC, m.id DESC
  LIMIT %(limit)s
)
-- ts_headline is the expensive part, so it runs on the LIMITed rows only. Fragment
-- mode suits long text but cuts a short message down to the bare matched word.
SELECT top.id, top.session_id, top.role, top.rank, s.title, s.started_at,
       CASE WHEN length(top.content) <= 160
            THEN ts_headline('simple', top.content, q.q, 'HighlightAll=true,StartSel=**,StopSel=**')
            ELSE ts_headline('simple', left(top.content, 20000), q.q,
                             'MaxFragments=1,MaxWords=20,MinWords=5,StartSel=**,StopSel=**')
       END AS snippet
FROM top JOIN sessions s ON s.id = top.session_id, q
ORDER BY top.rank DESC, top.id DESC
"""

_ILIKE_SQL = """
SELECT m.id, m.session_id, m.role, m.content, s.title, s.started_at
FROM messages m JOIN sessions s ON s.id = m.session_id
WHERE m.content ILIKE %(pattern)s
  AND m.active
  AND m.role = ANY(%(roles)s::text[])
  AND NOT (m.session_id = ANY(%(exclude)s::text[]))
  AND (%(include_subagents)s OR s.source <> %(subagent)s)
  AND (%(owner)s::text IS NULL OR s.owner_id = %(owner)s::text)
ORDER BY m.id DESC
LIMIT %(limit)s
"""


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _session(row: dict) -> SessionInfo:
    return SessionInfo(**row)


def _message(row: dict) -> StoredMessage:
    return StoredMessage(**row)


class PostgresSessionStore:
    def __init__(self, dsn: str, *, schema: str | None = None,
                 min_size: int = 1, max_size: int = 5) -> None:
        """``schema`` puts every table in that schema, created if missing
        (``get_default_store`` uses ``gg_agent``; tests use a throwaway one)."""
        if schema is not None and not _SCHEMA_NAME.match(schema):
            raise ValueError(f"invalid schema name {schema!r}")
        self.dsn, self.schema = dsn, schema
        self._min_size, self._max_size = min_size, max_size
        self._pool: AsyncConnectionPool | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def open(self) -> None:
        if self._pool is not None:
            return
        # client_encoding pinned: on a SQL_ASCII database psycopg would otherwise
        # hand text back as bytes.
        kwargs: dict = {"row_factory": dict_row, "autocommit": True, "client_encoding": "utf8"}
        if self.schema:
            # public stays on the path: that's where pg_trgm's operator classes live.
            kwargs["options"] = f"-c search_path={self.schema},public"
        # One plain connection first: an unreachable server fails here, at once and
        # with the driver's own message, instead of as a pool timeout after a
        # burst of logged retries.
        probe = await AsyncConnection.connect(self.dsn, connect_timeout=int(OPEN_TIMEOUT_SECONDS), **kwargs)
        try:
            if self.schema:
                # Must exist BEFORE migrating: search_path silently skips a missing
                # schema, and the tables would land in public instead.
                await probe.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
        finally:
            await probe.close()
        pool = AsyncConnectionPool(self.dsn, open=False, min_size=self._min_size,
                                   max_size=self._max_size, kwargs=kwargs)
        try:
            await pool.open(wait=True, timeout=OPEN_TIMEOUT_SECONDS)
            await migrate(pool)
        except BaseException:
            await pool.close()
            raise
        self._pool = pool

    async def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    @property
    def pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("PostgresSessionStore is not open — call `await store.open()` first")
        return self._pool

    async def _fetchall(self, sql: str, params=None) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall()

    # ── Users ────────────────────────────────────────────────────────────

    async def create_user(self, user_id, email, password_hash) -> UserInfo:
        try:
            rows = await self._fetchall(
                f"INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s) RETURNING {_USER_COLUMNS}",
                (user_id, email, password_hash))
        except errors.UniqueViolation:
            raise ValueError(f"email {email!r} is already registered") from None
        return UserInfo(**rows[0])

    async def get_user(self, user_id):
        rows = await self._fetchall(f"SELECT {_USER_COLUMNS} FROM users WHERE id = %s", (user_id,))
        return UserInfo(**rows[0]) if rows else None

    async def get_user_by_email(self, email):
        rows = await self._fetchall(f"SELECT {_USER_COLUMNS} FROM users WHERE email = %s", (email,))
        return UserInfo(**rows[0]) if rows else None

    # ── Writes ───────────────────────────────────────────────────────────

    async def create_session(self, session_id, *, owner_id, source, provider, model, system_prompt,
                             cwd, parent_session_id=None) -> None:
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO sessions (id, owner_id, source, provider, model, system_prompt, cwd, "
                    "parent_session_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (session_id, owner_id, source, provider, model, system_prompt, cwd, parent_session_id))
        except errors.ForeignKeyViolation:
            raise KeyError(f"no user {owner_id!r} (or no parent session {parent_session_id!r})") from None

    async def append_messages(self, session_id, messages) -> None:
        if not messages:
            return
        rows = [message_to_row(m) for m in messages]
        title = next((make_title(r["content"]) for r in rows if r["role"] == "user"), None)
        async with self.pool.connection() as conn, conn.transaction():
            # Serializes concurrent writers to ONE session; others don't wait.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
            # Counters first: a missing session is a KeyError (as in the in-memory
            # store), not a foreign-key violation, and the transaction undoes both.
            cur = await conn.execute(
                "UPDATE sessions SET message_count = message_count + %s, "
                "tool_call_count = tool_call_count + %s, last_activity_at = now(), "
                "title = coalesce(title, %s) WHERE id = %s",
                (len(rows), count_tool_calls(messages), title, session_id))
            if cur.rowcount == 0:
                raise KeyError(f"no session {session_id!r}")
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, tool_name) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [(session_id, r["role"], r["content"],
                      Jsonb(r["tool_calls"]) if r["tool_calls"] else None,
                      r["tool_call_id"], r["tool_name"]) for r in rows])

    async def add_usage(self, session_id, input_tokens, output_tokens) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE sessions SET input_tokens = input_tokens + %s, "
                "output_tokens = output_tokens + %s WHERE id = %s",
                (input_tokens or 0, output_tokens or 0, session_id))

    async def end_session(self, session_id, reason) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE sessions SET ended_at = now(), end_reason = %s "
                "WHERE id = %s AND ended_at IS NULL", (reason, session_id))

    # ── Reads ────────────────────────────────────────────────────────────

    async def get_session(self, session_id):
        rows = await self._fetchall(f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = %s", (session_id,))
        return _session(rows[0]) if rows else None

    async def get_messages(self, session_id):
        rows = await self._fetchall(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE session_id = %s AND active ORDER BY id",
            (session_id,))
        return [_message(r) for r in rows]

    async def load_history(self, session_id):
        return history_from_rows(await self.get_messages(session_id))

    async def list_sessions(self, *, limit=20, cwd=None, include_subagents=False, owner_id=None):
        where, params = [], []
        if owner_id is not None:
            where.append("owner_id = %s")
            params.append(owner_id)
        if cwd is not None:
            where.append("cwd = %s")
            params.append(cwd)
        if not include_subagents:
            where.append("source <> %s")
            params.append(SUBAGENT_SOURCE)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = await self._fetchall(
            f"SELECT {_SESSION_COLUMNS} FROM sessions {clause} "
            "ORDER BY last_activity_at DESC, started_at DESC LIMIT %s", (*params, limit))
        return [_session(r) for r in rows]

    async def search(self, query, *, limit=20, roles=("user", "assistant"),
                     exclude_session_ids=(), include_subagents=False, owner_id=None):
        query = (query or "").strip()
        if not query:
            return []
        params = {"q": query, "roles": list(roles), "exclude": list(exclude_session_ids),
                  "include_subagents": include_subagents, "subagent": SUBAGENT_SOURCE, "limit": limit,
                  "owner": owner_id}
        rows = await self._fetchall(_SEARCH_SQL, params)
        if rows:
            return [SearchHit(message_id=r["id"], session_id=r["session_id"], role=r["role"],
                              snippet=r["snippet"], title=r["title"], started_at=r["started_at"],
                              rank=float(r["rank"])) for r in rows]
        # Fallback: a raw substring, for what the tokenizer splits apart
        # (paths, identifiers with punctuation, partial words).
        needle = query.strip('"')
        rows = await self._fetchall(_ILIKE_SQL, {**params, "pattern": f"%{_escape_like(needle)}%"})
        return [SearchHit(message_id=r["id"], session_id=r["session_id"], role=r["role"],
                          snippet=snippet_around(r["content"], needle), title=r["title"],
                          started_at=r["started_at"]) for r in rows]

    async def messages_around(self, session_id, message_id, window=5):
        before = await self._fetchall(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE session_id = %s AND active AND id < %s "
            "ORDER BY id DESC LIMIT %s", (session_id, message_id, window))
        rest = await self._fetchall(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE session_id = %s AND active AND id >= %s "
            "ORDER BY id LIMIT %s", (session_id, message_id, window + 1))
        after = [r for r in rest if r["id"] != message_id][:window]
        anchor = [r for r in rest if r["id"] == message_id]
        return {"session_id": session_id, "anchor_id": message_id,
                "messages": [_message(r) for r in [*reversed(before), *anchor, *after]],
                "messages_before": len(before), "messages_after": len(after)}
