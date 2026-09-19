"""In-process ``SessionStore``: dicts and a counter.

Used by the offline tests and handy for embedding. It honours the same contract
as the Postgres store (tests/test_persistence.py runs one suite against both);
only search is simpler — every whitespace-separated term must appear in the
message, case-insensitively, newest first.
"""

from __future__ import annotations

import copy
import itertools
from datetime import datetime, timezone

from . import SUBAGENT_SOURCE, SearchHit, SessionInfo, StoredMessage, UserInfo, history_from_rows, make_title
from .serialize import count_tool_calls, message_to_row, snippet_around


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._users: dict[str, UserInfo] = {}
        self._sessions: dict[str, SessionInfo] = {}
        self._messages: list[StoredMessage] = []
        self._ids = itertools.count(1)
        self.opened = False

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.opened = False

    # ── Users ────────────────────────────────────────────────────────────

    async def create_user(self, user_id, email, password_hash) -> UserInfo:
        if any(u.email == email for u in self._users.values()):
            raise ValueError(f"email {email!r} is already registered")
        self._users[user_id] = UserInfo(id=user_id, email=email, password_hash=password_hash,
                                        created_at=_now())
        return copy.copy(self._users[user_id])

    async def get_user(self, user_id):
        user = self._users.get(user_id)
        return copy.copy(user) if user else None

    async def get_user_by_email(self, email):
        return next((copy.copy(u) for u in self._users.values() if u.email == email), None)

    # ── Sessions ─────────────────────────────────────────────────────────

    async def create_session(self, session_id, *, owner_id, source, provider, model, system_prompt,
                             cwd, parent_session_id=None) -> None:
        if session_id in self._sessions:
            raise ValueError(f"session {session_id!r} already exists")
        if owner_id not in self._users:          # the foreign key, as Postgres enforces it
            raise KeyError(f"no user {owner_id!r}")
        now = _now()
        self._sessions[session_id] = SessionInfo(
            id=session_id, owner_id=owner_id, source=source, parent_session_id=parent_session_id,
            provider=provider, model=model, system_prompt=system_prompt, cwd=cwd,
            started_at=now, last_activity_at=now)

    def _require(self, session_id: str) -> SessionInfo:
        info = self._sessions.get(session_id)
        if info is None:
            raise KeyError(f"no session {session_id!r}")
        return info

    async def append_messages(self, session_id, messages) -> None:
        info = self._require(session_id)
        now = _now()
        for msg in messages:
            row = message_to_row(msg)
            row["tool_calls"] = copy.deepcopy(row["tool_calls"])
            self._messages.append(StoredMessage(id=next(self._ids), session_id=session_id,
                                                created_at=now, **row))
            if info.title is None and msg.get("role") == "user":
                info.title = make_title(msg.get("content"))
        info.message_count += len(messages)
        info.tool_call_count += count_tool_calls(messages)
        info.last_activity_at = now

    async def add_usage(self, session_id, input_tokens, output_tokens) -> None:
        info = self._require(session_id)
        info.input_tokens += input_tokens or 0
        info.output_tokens += output_tokens or 0

    async def end_session(self, session_id, reason) -> None:
        info = self._sessions.get(session_id)
        if info is not None and info.ended_at is None:
            info.ended_at, info.end_reason = _now(), reason

    async def get_session(self, session_id):
        info = self._sessions.get(session_id)
        return copy.copy(info) if info else None

    async def get_messages(self, session_id):
        return [copy.copy(m) for m in self._messages if m.session_id == session_id]

    async def load_history(self, session_id):
        return history_from_rows(await self.get_messages(session_id))

    async def list_sessions(self, *, limit=20, cwd=None, include_subagents=False, owner_id=None):
        rows = [s for s in self._sessions.values()
                if (cwd is None or s.cwd == cwd)
                and (owner_id is None or s.owner_id == owner_id)
                and (include_subagents or s.source != SUBAGENT_SOURCE)]
        # started_at breaks ties: two sessions touched within one clock tick.
        rows.sort(key=lambda s: (s.last_activity_at, s.started_at), reverse=True)
        return [copy.copy(s) for s in rows[:limit]]

    async def search(self, query, *, limit=20, roles=("user", "assistant"),
                     exclude_session_ids=(), include_subagents=False, owner_id=None):
        terms = [t.strip('"').lower() for t in query.split() if t.strip('"')]
        if not terms:
            return []
        hits = []
        for m in reversed(self._messages):
            session = self._sessions[m.session_id]
            text = (m.content or "").lower()
            if (m.role in roles and m.session_id not in exclude_session_ids
                    and (include_subagents or session.source != SUBAGENT_SOURCE)
                    and (owner_id is None or session.owner_id == owner_id)
                    and all(t in text for t in terms)):
                hits.append(SearchHit(message_id=m.id, session_id=m.session_id, role=m.role,
                                      snippet=snippet_around(m.content, terms[0]),
                                      title=session.title, started_at=session.started_at))
                if len(hits) >= limit:
                    break
        return hits

    async def messages_around(self, session_id, message_id, window=5):
        rows = await self.get_messages(session_id)
        before = [m for m in rows if m.id < message_id][-window:]
        after = [m for m in rows if m.id > message_id][:window]
        anchor = [m for m in rows if m.id == message_id]
        return {"session_id": session_id, "anchor_id": message_id,
                "messages": before + anchor + after,
                "messages_before": len(before), "messages_after": len(after)}
