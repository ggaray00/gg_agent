"""Session persistence: transcripts that survive the process, resume, search.

The agent only ever talks to a ``SessionStore``. Two implementations:

  * ``InMemorySessionStore`` — dicts, no dependencies. The default in tests.
  * ``PostgresSessionStore`` — the real one (psycopg 3, async pool).

Persistence is opt-in: ``get_default_store()`` returns a Postgres store when
``GG_DATABASE_URL`` is set and ``None`` otherwise, and ``None`` means the agent
behaves exactly as it did before this package existed.

Every session has an owner: a row in ``users`` (email + scrypt password hash,
see ``users.py``). An agent without a ``user_id`` does not persist.

hermes-agent keeps the same data in SQLite, and most of its ~14k lines manage
SQLite-as-a-shared-file (WAL, read pools, corruption repair, FTS rebuilds). A
Postgres server has none of those problems, so what is left is small.

Mirrors hermes-agent: hermes_state*.py (the parts that aren't SQLite plumbing)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from .serialize import repair_for_resume, row_to_message

# Rows hidden from browsing and search by default: delegated children are the
# agent's own scratch work, not the user's history.
SUBAGENT_SOURCE = "subagent"
TITLE_MAX_CHARS = 80


@dataclass
class UserInfo:
    id: str
    email: str
    password_hash: str = field(default="", repr=False)
    created_at: datetime | None = None


@dataclass
class SessionInfo:
    id: str
    owner_id: str | None = None
    source: str = "cli"
    parent_session_id: str | None = None
    provider: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    cwd: str | None = None
    title: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    end_reason: str | None = None
    message_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    last_activity_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredMessage:
    """One ``messages`` row. ``id`` is the global ordering key — never sort by time."""

    id: int
    session_id: str
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    created_at: datetime | None = None

    def to_message(self) -> dict[str, Any]:
        """Back to the OpenAI-shaped dict the loop works with."""
        return row_to_message(self.__dict__)


@dataclass
class SearchHit:
    message_id: int
    session_id: str
    role: str
    snippet: str
    title: str | None = None
    started_at: datetime | None = None
    rank: float = 0.0


class SessionStore(Protocol):
    """What the agent, the CLI and the ``session_search`` tool need. Nothing more.

    Every method is async: the Postgres pool lives on the agent's event loop.
    ``open`` must be idempotent — a parent and its subagents share one store.
    """

    async def open(self) -> None: ...
    async def close(self) -> None: ...
    # Users. The store only keeps what it's given; hashing lives in users.py.
    async def create_user(self, user_id: str, email: str, password_hash: str) -> UserInfo: ...
    async def get_user(self, user_id: str) -> UserInfo | None: ...
    async def get_user_by_email(self, email: str) -> UserInfo | None: ...
    # Sessions. ``owner_id=None`` on a read means "every owner" (admin use);
    # the agent, CLI and search tool always pass the signed-in user.
    async def create_session(self, session_id: str, *, owner_id: str, source: str,
                             provider: str | None, model: str | None, system_prompt: str | None,
                             cwd: str | None, parent_session_id: str | None = None) -> None: ...
    async def append_messages(self, session_id: str, messages: list[dict]) -> None: ...
    async def add_usage(self, session_id: str, input_tokens: int, output_tokens: int) -> None: ...
    async def end_session(self, session_id: str, reason: str) -> None: ...
    async def get_session(self, session_id: str) -> SessionInfo | None: ...
    async def get_messages(self, session_id: str) -> list[StoredMessage]: ...
    async def load_history(self, session_id: str) -> list[dict]: ...
    async def list_sessions(self, *, limit: int = 20, cwd: str | None = None,
                            include_subagents: bool = False,
                            owner_id: str | None = None) -> list[SessionInfo]: ...
    async def search(self, query: str, *, limit: int = 20, roles=("user", "assistant"),
                     exclude_session_ids=(), include_subagents: bool = False,
                     owner_id: str | None = None) -> list[SearchHit]: ...
    async def messages_around(self, session_id: str, message_id: int, window: int = 5) -> dict: ...


def make_title(content: Any) -> str | None:
    """A session's title is its first user message, collapsed onto one line."""
    if not isinstance(content, str):
        return None
    text = " ".join(content.split())
    if not text:
        return None
    return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1] + "…"


def history_from_rows(rows: list[StoredMessage]) -> list[dict]:
    """Rows in id order → a transcript the next provider request will accept."""
    return repair_for_resume([r.to_message() for r in rows])


DEFAULT_SCHEMA = "gg_agent"


def get_default_store() -> SessionStore | None:
    """Postgres when ``GG_DATABASE_URL`` is set, otherwise no persistence at all.

    Tables live in their own schema (``$GG_DATABASE_SCHEMA``, default ``gg_agent``)
    so they can share a database with other apps' ``users`` / ``sessions``.
    """
    dsn = os.getenv("GG_DATABASE_URL", "").strip()
    if not dsn:
        return None
    from .postgres import PostgresSessionStore

    return PostgresSessionStore(dsn, schema=os.getenv("GG_DATABASE_SCHEMA", "").strip() or DEFAULT_SCHEMA)


from .memory import InMemorySessionStore  # noqa: E402  (needs the dataclasses above)

__all__ = [
    "SessionStore", "SessionInfo", "SearchHit", "StoredMessage", "UserInfo",
    "DEFAULT_SCHEMA",
    "InMemorySessionStore", "get_default_store", "make_title", "history_from_rows",
    "SUBAGENT_SOURCE",
]
