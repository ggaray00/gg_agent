# Plan: Postgres session persistence — transcript, resume, search

Branch: `dev-db-interation` · Status: implemented 2026-09-19 · Written 2026-09-19

This plan is self-contained: a fresh session should be able to implement it
without the conversation that produced it. Read "Context" and "Decisions" first,
then work the phases in order — each ends in a green `uv run pytest`.

---

## Context

gg-agent has **no persistence today**. `Agent.history` is an in-memory list,
`Agent.session_id` is a throwaway `uuid4().hex[:12]`, and a process exit loses
everything.

The reference implementation is hermes-agent's SQLite store
(`/Users/ggaray/OtherPProjects/hermes-agent/hermes_state*.py`, ~14.5k lines).
Only a small part of it is worth porting; most of it manages SQLite-as-a-shared-file
(WAL, read pools, file-holder scans, corruption repair, FTS5 rebuilds) — problems
a Postgres server simply doesn't have.

Useful hermes files to consult (read-only reference):

| What | Where |
|---|---|
| Schema (sessions, messages) | `hermes_state_common.py:269-370` |
| Append a message / batch | `hermes_state_messages.py:265-315` |
| Resume (tip resolution, model vs display history) | `hermes_state_messages.py:691-900` |
| Search engine | `hermes_state_search.py:1000-1110` |
| **The model-facing search tool — port its ideas** | `tools/session_search_tool.py` (whole file, 680 lines) |
| Flush-to-DB from the agent (dedup by marker) | `agent/session_persistence.py:313-395` |
| System-prompt nudge | `agent/prompt_builder.py:204` |

gg-agent files this plan touches: `gg_agent/loop.py`, `gg_agent/agent.py`,
`gg_agent/cli.py`, `gg_agent/prompts.py`, `gg_agent/tools/delegate_tool.py`,
`pyproject.toml`, `README.md`, plus new files below.

---

## Decisions (already made — don't re-litigate)

1. **Postgres via `psycopg` v3, async** (`psycopg[binary,pool]`, `AsyncConnectionPool`).
   The core is async-first; the pool must be opened on the loop that uses it —
   the same constraint MCP has, and `gg_agent/aio.py`'s persistent background loop
   already satisfies it for sync callers. Open the pool lazily inside the first
   `arun`, never in `Agent.__init__` (which is sync).
2. **Keyword search only**: a generated `tsvector` column + GIN index, and a
   `pg_trgm` index for substring fallback. **No embeddings / pgvector** in this
   plan (possible later; see "Out of scope").
3. **A `SessionStore` protocol with two implementations**: `InMemorySessionStore`
   (default for tests, zero deps at runtime) and `PostgresSessionStore`. The 26
   existing offline tests must keep passing with no database available.
4. **Persistence is opt-in**: enabled when `GG_DATABASE_URL` is set or a `store=`
   is passed to `Agent`. Without it, behaviour is exactly today's.
5. **A persistence failure never kills a turn.** Log a warning, emit a
   `persist_error` event, keep the in-memory history, retry the unflushed tail at
   the next flush point.
6. **The system prompt is not stored as a message.** `run_conversation` rebuilds it
   every turn; store it on the `sessions` row for reference only.
7. **Migrations** are numbered `.sql` files applied by a ~30-line `migrate()` that
   reads/writes a `schema_version` table. No Alembic.
8. **Search is pulled by the model, not pushed**: a `session_search` tool plus one
   line of system-prompt guidance, like hermes. Nothing is auto-injected per turn.

---

## Target layout

```
gg_agent/persistence/
    __init__.py          # SessionStore Protocol, SessionInfo/SearchHit dataclasses, get_default_store()
    memory.py            # InMemorySessionStore
    postgres.py          # PostgresSessionStore
    migrate.py           # apply migrations/*.sql in order
    serialize.py         # message dict <-> row conversion (shared by both stores)
    migrations/
        001_init.sql
gg_agent/tools/session_tools.py     # session_search tool (self-registers)
.env.example                        # GG_DATABASE_URL=...
tests/test_persistence.py           # offline, InMemorySessionStore
tests/test_persistence_pg.py        # @pytest.mark.pg, skipped without GG_TEST_DATABASE_URL
tests/test_session_search.py        # offline tool tests
```

Make sure `migrations/*.sql` ships with the package: add
`[tool.setuptools.package-data] "gg_agent.persistence" = ["migrations/*.sql"]`
to `pyproject.toml`.

---

## Phase 0 — Setup

- `uv add "psycopg[binary,pool]>=3.2"`
- **No Docker.** Use a Postgres server that is already running (local install such
  as Homebrew `postgresql@16`, or any reachable server). One-time setup:
  ```bash
  createuser gg --pwprompt            # password: gg
  createdb -O gg gg_agent
  createdb -O gg gg_agent_test
  psql -d gg_agent      -c 'CREATE EXTENSION IF NOT EXISTS pg_trgm'
  psql -d gg_agent_test -c 'CREATE EXTENSION IF NOT EXISTS pg_trgm'
  ```
  The extension is created here by a superuser because `gg` may lack the privilege;
  `001_init.sql` keeps `CREATE EXTENSION IF NOT EXISTS`, which is then a no-op.
- `.env.example`: `GG_DATABASE_URL=postgresql://gg:gg@localhost:5432/gg_agent`
  and `GG_TEST_DATABASE_URL=postgresql://gg:gg@localhost:5432/gg_agent_test`.
- `pyproject.toml`: register the marker
  `markers = ["pg: needs a live Postgres (GG_TEST_DATABASE_URL)"]`.

**Done when:** `psql "$GG_DATABASE_URL" -c 'select 1'` succeeds; `uv run pytest` still green.

---

## Phase 1 — Schema, store protocol, in-memory store

### `migrations/001_init.sql`

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS schema_version (version int NOT NULL);

CREATE TABLE sessions (
  id                text PRIMARY KEY,
  parent_session_id text REFERENCES sessions(id) ON DELETE SET NULL,
  source            text NOT NULL DEFAULT 'cli',     -- 'cli' | 'api' | 'subagent'
  provider          text,
  model             text,
  system_prompt     text,
  cwd               text,
  title             text,                            -- first user message, truncated to 80 chars
  metadata          jsonb NOT NULL DEFAULT '{}',
  started_at        timestamptz NOT NULL DEFAULT now(),
  ended_at          timestamptz,
  end_reason        text,
  message_count     int    NOT NULL DEFAULT 0,
  tool_call_count   int    NOT NULL DEFAULT 0,
  input_tokens      bigint NOT NULL DEFAULT 0,
  output_tokens     bigint NOT NULL DEFAULT 0,
  last_activity_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON sessions (parent_session_id);
CREATE INDEX ON sessions (last_activity_at DESC);
CREATE INDEX ON sessions (cwd, last_activity_at DESC);    -- for --continue

CREATE TABLE messages (
  id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,   -- THE ordering key
  session_id   text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role         text NOT NULL,                  -- user | assistant | tool
  content      text,
  tool_calls   jsonb,                          -- assistant only, OpenAI shape
  tool_call_id text,                           -- tool only
  tool_name    text,                           -- tool only (the message's "name" key)
  active       boolean NOT NULL DEFAULT true,  -- reserved for future compaction/rewind
  created_at   timestamptz NOT NULL DEFAULT now(),
  -- Capped: to_tsvector errors above ~1MB, and huge tool output is noise anyway.
  search_tsv   tsvector GENERATED ALWAYS AS (
                 to_tsvector('simple'::regconfig,
                   left(coalesce(content, ''), 100000) || ' ' || coalesce(tool_name, ''))) STORED
);
CREATE INDEX ON messages (session_id, id);
CREATE INDEX messages_fts  ON messages USING gin (search_tsv);
CREATE INDEX messages_trgm ON messages USING gin (content gin_trgm_ops)
  WHERE role IN ('user', 'assistant');
```

Notes:
- Order transcripts by `id`, never by `created_at` (hermes learned this: timestamps
  are not monotonic and break tool-call adjacency).
- `'simple'` text config = no stemming, which suits code identifiers and paths.
- `active` is unused in this plan but keeps the door open for compaction.

### `persistence/__init__.py`

```python
class SessionStore(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def create_session(self, session_id: str, *, source: str, provider: str | None,
                             model: str | None, system_prompt: str | None, cwd: str | None,
                             parent_session_id: str | None = None) -> None: ...
    async def append_messages(self, session_id: str, messages: list[dict]) -> None: ...
    async def add_usage(self, session_id: str, input_tokens: int, output_tokens: int) -> None: ...
    async def end_session(self, session_id: str, reason: str) -> None: ...
    async def get_session(self, session_id: str) -> SessionInfo | None: ...
    async def load_history(self, session_id: str) -> list[dict]: ...   # active rows, ORDER BY id
    async def list_sessions(self, *, limit: int = 20, cwd: str | None = None,
                            include_subagents: bool = False) -> list[SessionInfo]: ...
    async def search(self, query: str, *, limit: int = 20, roles=("user", "assistant"),
                     exclude_session_ids=(), include_subagents: bool = False) -> list[SearchHit]: ...
    async def messages_around(self, session_id: str, message_id: int, window: int = 5) -> dict: ...
```

`SessionInfo` / `SearchHit` are plain dataclasses. `get_default_store()` returns a
`PostgresSessionStore(os.environ["GG_DATABASE_URL"])` when that is set, else `None`.

### `persistence/serialize.py`

- `message_to_row(msg) -> dict`: `role`, `content` (str; if not a str, `json.dumps`
  it), `tool_calls` (list → jsonb), `tool_call_id`, `tool_name = msg.get("name")`.
- `row_to_message(row) -> dict`: the exact inverse, producing the same OpenAI-shaped
  dicts `loop.py` builds (`record_assistant_message`, `run_tool_round`). Omit keys
  that are `None` so a round-trip is byte-identical to what the loop produced.
- `repair_for_resume(messages) -> list[dict]`: if the transcript ends with an
  `assistant` message whose `tool_calls` lack matching `tool` replies (crash
  mid-tool-round), drop that dangling tail. Every provider rejects an unanswered
  tool call. (hermes: `_drop_trailing_empty_response_scaffolding`.)

### `persistence/memory.py`

Dicts + a counter for message ids. `search` = case-insensitive substring match
over `content`, newest first — good enough for tests. It must satisfy the same
contract tests as the Postgres store (see Phase 6).

**Done when:** `InMemorySessionStore` passes round-trip, ordering, resume-repair
and search unit tests.

---

## Phase 2 — Postgres store

`persistence/postgres.py`, `PostgresSessionStore(dsn)`:

- `open()`: create `AsyncConnectionPool(dsn, open=False, min_size=1, max_size=5)`,
  `await pool.open()`, then `await migrate(pool)`.
- `append_messages`: one transaction; first
  `SELECT pg_advisory_xact_lock(hashtext(%s))` on the session id (serializes
  concurrent writers to one session — replaces hermes's compression locks and turn
  leases), then `executemany` INSERT, then bump `message_count`, `tool_call_count`,
  `last_activity_at`; set `title` from the first user message if still NULL.
- `load_history`: `SELECT ... FROM messages WHERE session_id=%s AND active ORDER BY id`,
  map through `row_to_message`, then `repair_for_resume`.
- `search`:
  ```sql
  SELECT m.id, m.session_id, m.role, s.title, s.started_at,
         ts_headline('simple', m.content, q, 'MaxFragments=1,MaxWords=20,MinWords=5') AS snippet,
         ts_rank_cd(m.search_tsv, q) AS rank
  FROM messages m
  JOIN sessions s ON s.id = m.session_id,
       websearch_to_tsquery('simple', %(q)s) q
  WHERE m.search_tsv @@ q
    AND m.role = ANY(%(roles)s)
    AND NOT (m.session_id = ANY(%(exclude)s))
    AND (%(include_subagents)s OR s.source <> 'subagent')
  ORDER BY rank DESC, m.id DESC
  LIMIT %(limit)s;
  ```
  `websearch_to_tsquery` accepts `"phrases"`, `OR`, `-term` and never raises on user
  input — no sanitizer needed (hermes needs ~40 lines for FTS5).
  **Fallback** when that returns zero rows: `content ILIKE %s` with `%`/`_`/`\`
  escaped, backed by the trigram index; snippet = ±80 chars around the match.
- `messages_around`: rows with `id` in the ±window around the anchor inside that
  session, plus `messages_before` / `messages_after` counts.

**Done when:** `uv run pytest -m pg` (with `GG_TEST_DATABASE_URL` set and the
local server running) passes the shared contract tests against Postgres.

---

## Phase 3 — Loop and Agent integration

### `loop.py`

- Add to `LoopState`: `persist: Callable[[list[dict]], Awaitable[None]] | None = None`
  and `persisted_upto: int = 0` (index into `s.messages` of the first unflushed message).
- In `run_conversation`, after building `messages`, set
  `s.persisted_upto = len(messages) - 1` — i.e. system + prior history are already
  durable; the new user message is the first unflushed one.
- Add `async def flush(s)`: if `s.persist`, call it with
  `s.messages[s.persisted_upto:]` excluding any `role == "system"`; on success
  advance `persisted_upto`; on exception log + `agent._emit("persist_error", ...)`
  and leave `persisted_upto` unchanged (retried at the next flush).
- Flush points:
  1. right after `record_assistant_message` (persists user + assistant/tool_calls
     **before** tools run — the crash-safe point `run_tool_round`'s docstring invariant #1 describes);
  2. right after `run_tool_round` appends tool replies;
  3. once in the finalize block (catches interrupted / api_error / max_iterations exits).
- `run_conversation` gains a `persist=None` keyword it copies onto the state.
  The loop stays store-agnostic: it only knows a callback.

Index-based dedup is safe **only because gg-agent history is append-only**. When
compression or history repair is added later, switch to hermes's per-message
marker (`_DB_PERSISTED_MARKER` in `agent/session_persistence.py`).

### `agent.py`

- New `__init__` kwargs: `store: SessionStore | None | bool = None`
  (`None` → `get_default_store()`, `False` → disabled), `resume: str | None = None`,
  `source: str = "cli"`.
- Subagents inherit `parent.store` (shared pool) and use `source="subagent"`.
- Lazy start in `arun` (next to the existing MCP lazy connect), guarded by a
  `_store_ready` flag:
  1. `await store.open()` if this agent owns the store (not a child);
  2. if `resume`: `self.session_id = resume`; `self.history = await store.load_history(resume)`;
     raise `ValueError` if the session doesn't exist;
  3. else `await store.create_session(self.session_id, source=..., provider=self.profile.name,
     model=self.model, system_prompt=self.system_prompt, cwd=self.cwd,
     parent_session_id=self.parent.session_id if self.parent else None)`.
- Pass `persist=lambda msgs: store.append_messages(self.session_id, msgs)` into
  `run_conversation`; after the turn, `await store.add_usage(...)` from `result["usage"]`
  (failures logged, not raised).
- `reset()`: with a store, start a **new** session id (and end the old one with
  `end_reason="new_session"`) instead of just clearing the list.
- `aclose()`: `end_session(self.session_id, "closed")`, then close the store only if
  this agent opened it.
- Add `async def aresume(self, session_id)` / sync `resume()` for switching sessions
  in-process (used by the REPL).

### `tools/delegate_tool.py`

`_build_child` passes `store=parent_agent.store` and `source="subagent"`
(`parent=` is already passed, which supplies `parent_session_id`).

**Done when:** a FakeTransport test (copy the pattern from `tests/test_loop.py`)
with `InMemorySessionStore` shows: user, assistant(tool_calls), tool, assistant
rows in order; a second `Agent(resume=id)` reproduces the same history; an
exception raised from the store doesn't fail the turn and emits `persist_error`.

---

## Phase 4 — CLI

In `gg_agent/cli.py`:

- `--resume ID` → `Agent(resume=ID)`
- `--continue` → most recent non-subagent session for the current `cwd`
- `--sessions` → print recent sessions (id, when, title, message count) and exit
- `--search QUERY` → print hits (session id, role, snippet) and exit
- `--no-persist` → `Agent(store=False)`
- REPL commands: `/sessions`, `/resume ID`, `/new` (= reset → new session),
  `/search QUERY`. Keep `/history` and `/reset` working.
- On start, when persistence is on, print the session id to stderr next to the
  existing `[provider/model · N tools]` banner so the user can resume later.
- If `GG_DATABASE_URL` is set but the DB is unreachable, print one warning and run
  without persistence — never refuse to start.

**Done when:** manual run — start a REPL, ask two questions, exit, then
`uv run gg-agent --continue "what did I ask first?"` answers correctly.

---

## Phase 5 — `session_search` tool

`gg_agent/tools/session_tools.py`, self-registering like `delegate_tool.py`:
`registry.register_toolset("sessions", ...)`, `registry.register(name="session_search",
toolset="sessions", needs_agent=True, check_fn=<store configured?>, ...)`.
The handler receives `parent_agent` and uses `parent_agent.store` and
`parent_agent.session_id`.

Port these ideas from hermes `tools/session_search_tool.py`:

1. **Four modes, picked by the arguments**
   - `query` → **discover**: top N sessions (default 3, max 10)
   - `session_id` + `around_message_id` → **scroll**: ±`window` messages (default 5, clamp 1–20)
   - `session_id` alone → **read**: first 20 + last 10 messages
   - no args → **browse**: recent sessions
2. **Discover pipeline**: over-fetch 100 hits → drop the current session (its content
   is already in context) and subagent sessions → keep the best hit per session
   up to `limit` → **adaptive detail**: the top session gets a ±5 window around its
   hit (each message capped at 4000 chars), the rest get only the matched message + snippet.
3. **Default `roles` = user + assistant**; `role_filter` can opt into `tool`.
4. **Coaching in responses**:
   - zero results → explain the syntax: quoted phrases, `OR`, `-term`
   - every discover result → hint to scroll with `session_id` + `around_message_id=match_message_id`
   - scroll → hint that `messages_before/after < window` means the end was reached
5. **Schema description** says it searches past conversation history only and must
   not be used to conclude something doesn't exist — check files/live sources first.
6. **Returns real DB messages, no LLM calls.**

In `prompts.py`, add one guidance line to the main system prompt **only when the
tool is available** (pass a flag into `build_system_prompt`):

> When the user refers to something from a past conversation, or you suspect
> relevant context exists in an earlier session, use session_search to recall it
> before asking them to repeat themselves.

Subagents do **not** get `session_search` by default (add it to the child's
`blocked_tools` in `_build_child`) — they work from the context they were handed.

**Done when:** offline tests cover all four modes, current-session exclusion,
per-session dedup, adaptive detail, and the zero-results message.

---

## Phase 6 — Tests

- `tests/test_persistence.py` (offline): a **store contract test** function
  parametrized over store factories — `InMemorySessionStore` always,
  `PostgresSessionStore` added only when `GG_TEST_DATABASE_URL` is set. The same
  assertions run against both. Cover: create/append/load round-trip is
  byte-identical, ordering by id, `repair_for_resume` drops a dangling tool call,
  counters and title, `list_sessions` filters (cwd, subagents), search roles and
  exclusions.
- `tests/test_persistence_pg.py` (`@pytest.mark.pg`): Postgres-specific behaviour —
  migrations are idempotent (run twice), `websearch_to_tsquery` syntax (`"a b"`,
  `x OR y`, `-z`), ILIKE fallback with `%` in the query, a 2MB tool output inserts
  fine (tsvector cap), concurrent `append_messages` to one session keeps ids ordered.
  Each test uses a fresh schema or truncates tables in a fixture.
- Loop-level test (offline) as described in Phase 3.
- `tests/test_session_search.py` (offline) as described in Phase 5.

Commands:

```bash
uv run pytest                                  # offline suite, no DB needed
GG_TEST_DATABASE_URL=postgresql://gg:gg@localhost:5432/gg_agent_test uv run pytest -m pg
uv run ruff check gg_agent/ tests/
```

---

## Phase 7 — Docs

- README: a "Persistence" section — enable with `GG_DATABASE_URL`, the one-time
  `createuser`/`createdb`/`pg_trgm` setup from Phase 0,
  the CLI flags, the REPL commands, and a note that it is off by default.
- Module docstrings in the repo's style: each new file names the hermes file it
  was distilled from (e.g. `Mirrors hermes-agent: hermes_state_messages.py + agent/session_persistence.py`).
- Update the test count in the README quickstart.

---

## Out of scope (future, don't build now)

- Context compression / compaction (the `active` column is reserved for it), rewind, branching.
- Semantic search / pgvector. If keyword search proves insufficient, add a
  `vector` column on `messages` for user + final assistant rows only and merge
  rankings with reciprocal rank fusion — a migration `00N_*.sql`, not a new system.
- Per-model usage table, cost accounting, LLM-generated titles.
- Multi-user / auth, retention pruning, export/import.

## Acceptance checklist

- [ ] `uv run pytest` green with no database and no env vars
- [ ] `uv run pytest -m pg` green against the local Postgres server
- [ ] `uv run ruff check gg_agent/ tests/` clean
- [ ] Without `GG_DATABASE_URL`, behaviour and output are unchanged from `dev`
- [ ] Kill the process mid-tool-round → `--resume` loads a valid transcript (dangling call dropped)
- [ ] `--continue`, `--sessions`, `--search` and the REPL commands work end to end
- [ ] The model uses `session_search` to answer "what did we decide about X last time?"
