# gg-agent

A bare-bones, runnable reconstruction of [hermes-agent](https://github.com/)'s core:
**pick an LLM provider → run an agentic loop → call tools → spawn subagents.**

hermes-agent is ~1.5M lines. This is ~1,700 (plus ~800 of tests), and it is the same architecture, not a
different one — every file below names the hermes file it was distilled from, so you
can reconstruct the real thing one layer at a time.

## Quickstart (uv)

```bash
uv sync                                  # creates .venv + installs from uv.lock

uv run gg-agent --auth-status            # where is each provider's credential coming from?
uv run gg-agent "how many python files are here, and which is largest?"
uv run gg-agent                          # interactive REPL
uv run gg-agent --list-providers --list-tools
uv run gg-agent --list-models            # live catalog for the active provider
uv run python example_subagents.py       # delegation, two ways

uv run pytest                            # offline tests, no keys or database needed
uv run ruff check gg_agent/
```

Credentials, in the order `gg-agent` looks for them:

| Provider | How to authenticate |
|---|---|
| **GitHub Copilot** | already signed in via VS Code / Copilot CLI → **nothing to do**; else `uv run gg-agent --login` |
| anthropic / openai / openrouter / groq / deepseek | `export ANTHROPIC_API_KEY=…` (etc.) |
| ~40 more API-key providers (xai, gemini, zai, kimi, minimax, fireworks, nebius, …) | their env var — `--list-providers` / `--auth-status` name it; auto-detected after the ones above |
| openai-codex | a ChatGPT subscription: sign in once with `codex login`, then `-p codex` |
| bedrock / vertex | the AWS / Google Cloud credential chain (`pip install boto3` / `google-auth`), then `-p bedrock` / `-p vertex` |
| copilot-acp | the Copilot CLI on `PATH` (`npm install -g @github/copilot`), then `-p copilot-acp` |
| ollama / custom | nothing — `-p ollama`, or `CUSTOM_BASE_URL=… -p custom` for any OpenAI-compatible server |

Every provider accepts `-r/--reasoning low|medium|high|xhigh|max|off` (or `GG_REASONING`);
each profile translates it to that provider's own knob (`extra_body.reasoning`,
`reasoning_effort`, `thinking`, Gemini's `thinking_config`, a Responses `reasoning.effort`…),
clamped to the levels that model accepts. Unset = the provider's default.

No install step needed for one-off runs — `uv run run.py …` and `uv run python -m gg_agent …`
work the same way. To add a dependency: `uv add <pkg>`.

```python
from gg_agent import Agent

with Agent(provider="copilot", model="gpt-4.1") as agent:
    print(agent.ask("Summarize what this repo does."))
```

The core is **async**; `ask`/`run`/`close` are sync wrappers over a background
event loop so scripts and tests need no ceremony. When you are already async:

```python
async with Agent(provider="copilot") as agent:
    print(await agent.aask("Summarize what this repo does."))
```

Calling a sync wrapper from inside a running loop raises rather than deadlocking.

**`scripts/`** is a folder of runnable, no-arguments examples — open one, edit the
constants at the top, press Run. Start with `scripts/00_check_setup.py`, which
makes no API call.

---

## The whole idea

```
run.py / cli.py                 ← front-end
     │
  Agent (agent.py)              ← resolves provider, owns client + tool grant + history
     │
run_conversation (loop.py)      ← THE LOOP
     │
     ├─ transport.build_kwargs ─────────► OpenAI msgs → provider-native wire format
     ├─ await transport.call ───────────► the HTTP request
     ├─ transport.normalize_response ───► provider response → NormalizedResponse
     └─ await registry.dispatch ────────► run the tools the model asked for
                                             ├─ built-in tools (files, shell)
                                             ├─ MCP tools  → mcp_tools.py → a server
                                             └─ delegate_task → child Agent → (recurse)
```

Everything below `Agent` is async. `aio.py` is the single sync↔async boundary, so
`agent.run(...)` from a script and `await agent.arun(...)` from an async caller
reach the same loop.

The loop is deliberately small enough to read in one sitting:

```python
while budget remains:
    assemble request  ->  call model  ->  normalize response
    if the model asked for tools:  run them, append results, loop again
    else:                          that text is the answer, stop
```

Everything a production agent adds — compression, failover, checkpoints, approval
gates, streaming, persistence — hangs off those four phases without changing them.

---

## Three invariants worth internalizing

1. **The loop speaks one message format.** History is always OpenAI-shaped
   (`role`/`content`/`tool_calls`/`tool_call_id`). Only the transport knows what
   Anthropic or any other protocol actually wants on the wire. Adding a provider
   never touches the loop.
2. **Tool failures go back to the model, not up the stack.** A raised exception in a
   tool would end the turn; an error string lets the model fix the path and retry.
   `registry.dispatch` catches everything.
3. **Every tool call gets exactly one `role:"tool"` reply, in request order.** A
   missing or reordered reply makes the *next* request invalid on every provider.
   This is the single most common way a hand-rolled agent loop breaks.

---

## File map — gg_agent → hermes-agent

| gg_agent | what it does | distilled from |
|---|---|---|
| `providers/base.py` | `ProviderProfile`: one provider declared once (auth, endpoint, quirks, hooks) | `providers/base.py` |
| `providers/__init__.py` | registry, plugin discovery, auto-detect order | `providers/__init__.py` |
| `providers/plugins/*.py` | ~50 profiles, one vendor per module (also loads `$GG_HOME/plugins/model-providers/*.py`) | `plugins/model-providers/*` |
| `providers/copilot_acp_client.py` | Copilot CLI over ACP (stdio JSON-RPC) behind an OpenAI-client face | `agent/copilot_acp_client.py`, `agent/acp_openai_bridge.py` |
| `reasoning_effort.py` | effort ladder + per-wire vocabularies + `clamp_effort` | `agent/reasoning_effort.py` |
| `providers/copilot_auth.py` | Copilot OAuth: token discovery, exchange, caching, device login | `hermes_cli/copilot_auth.py` |
| `transports/types.py` | `ToolCall` / `Usage` / `NormalizedResponse` — the only types the loop sees | `agent/transports/types.py` |
| `transports/base.py` | `ProviderTransport` ABC: convert → build → call → normalize | `agent/transports/base.py` |
| `transports/chat_completions.py` | OpenAI + every OpenAI-compatible endpoint | `agent/transports/chat_completions.py` |
| `transports/anthropic.py` | Messages API: system extraction, `tool_use`/`tool_result` blocks, cache breakpoints, Bearer auth | `agent/transports/anthropic.py` + `agent/prompt_caching.py` |
| `transports/responses.py` | Responses API (xAI, Meta, Router, Actual, Codex): input items, `function_call`s, stream-only backends | `agent/transports/codex.py`, `agent/codex_responses_adapter.py` |
| `transports/bedrock.py` | AWS Bedrock Converse via boto3: `toolUse`/`toolResult`, cache points | `agent/transports/bedrock.py`, `agent/bedrock_adapter.py` |
| `transports/streaming.py` | `StreamHooks`, tool-call delta reassembly, stream errors | `agent/chat_completion_helpers.py` (`_StreamingCall`, `_ToolCallAccumulator`) |
| `stream_delivery.py` | what reaches the screen: `<think>` scrubbing, segment breaks, stream events | `agent/stream_delivery.py`, `agent/think_scrubber.py` |
| `tools/registry.py` | `ToolEntry`, registration, toolset filtering, safe dispatch | `tools/registry.py` + `model_tools.handle_function_call` |
| `tools/shell_tool.py` | `run_shell` | `tools/terminal_tool*.py` |
| `tools/file_tools.py` | `read_file` / `write_file` / `list_dir` | `tools/file_tools.py`, `file_operations_*.py` |
| `tools/delegate_tool.py` | `delegate_task`: parallel subagents, depth cap, timeouts | `tools/delegate_tool*.py` (7 modules) |
| `tools/mcp_tools.py` | MCP servers registered as ordinary tools: config, lifecycle, translation | `plugins/mcp/*` |
| `aio.py` | the one sync↔async boundary: a persistent background loop | — |
| `loop.py` | `run_conversation` + `LoopState` + phases + parallel tool execution | `agent/conversation_loop.py`, `agent/turn_*.py`, `agent/tool_executor.py` |
| `compression.py` | keeping a long session inside the context window: sizing, thresholds, pruning, summarizing | `agent/context_compressor.py` |
| `agent.py` | `Agent`: provider resolution, client, tool grant, interrupts, turn facade | `run_agent.AIAgent`, `agent/agent_init.py`, `agent/client_lifecycle.py` |
| `prompts.py` | main + child system prompts | `agent/prompt_builder.py`, `tools/delegate_tool_progress.py` |
| `cli.py` | one-shot / REPL front-end, event rendering | `cli.py` |
| `persistence/__init__.py` | `SessionStore` protocol, `SessionInfo` / `SearchHit`, `get_default_store()` | `hermes_state*.py` (the non-SQLite parts) |
| `persistence/postgres.py` | Postgres store: advisory-locked appends, `tsvector` + trigram search | `hermes_state_messages.py`, `hermes_state_search.py` |
| `persistence/memory.py` | in-process store with the same contract (tests, embedding) | — |
| `persistence/serialize.py` | message dict ↔ row, resume repair of a dangling tool round | `hermes_state_messages.py` |
| `persistence/users.py` | email + password users (stdlib scrypt hashing), register / authenticate | — |
| `persistence/migrate.py` | numbered `migrations/*.sql`, a `schema_version` table, no Alembic | — |
| `tools/session_tools.py` | `session_search`: discover / scroll / read / browse past sessions | `tools/session_search_tool.py` |

**The `LoopState` + phase-function shape in `loop.py` is not decoration.** It is exactly
how hermes scales that loop: each phase (`assemble_request`, `perform_api_call`,
`run_tool_round`, …) reads the state fields it needs and rebinds the ones it owns, so a
phase can move into its own module without changing a signature. hermes has ~20 such
`agent/turn_*.py` modules threading one dataclass.

---

## Prompt caching

On by default. Caching is a **prefix match** — the key is the exact bytes of the
rendered prompt up to each breakpoint — and the render order is
`tools` → `system` → `messages`. Two of the four breakpoints the API allows cover an
agent loop: one on the system block, which also caches the tool definitions behind it,
and one on the last message, so the next iteration reads back everything before it.
Cache reads cost ~0.1× input price, writes 1.25×, so a turn breaks even on its second
API call.

Only `transports/anthropic.py` asks for this explicitly — `cache_control` is Anthropic's
parameter. Every OpenAI-compatible endpoint (OpenAI, Groq, DeepSeek, OpenRouter) caches
prefixes on its own with no request parameter, and `transports/chat_completions.py`
swallows the flag rather than forwarding it. What pays off on *all* of them is the
prefix discipline the markers force: tools serialized in a stable order
(`tools/registry.py` sorts by name), a system prompt built once per session rather than
per request, and history that is appended to, never rewritten.

```bash
uv run gg-agent --no-prompt-cache "…"     # or Agent(prompt_caching=False)
```

`Usage` reports the split: `cached_tokens` (served from cache) and `cache_write_tokens`
(written to it) are both parts of `prompt_tokens`, never additions to it. Anthropic's
`input_tokens` is the *uncached remainder*, so the transport adds the cache fields back
before reporting — otherwise a long cached session under-reports its own prompt size.
Watch those fields after any change to prompt assembly: caching fails silently, and a
broken prefix looks exactly like a working one except on the bill.

---

## Context compression

Every session eventually runs out of window. Three passes, cheapest first, each one
run only because the one before it left the request still over the threshold:

- **A — reclaim, no LLM.** Old tool results become one-line stubs, identical results
  collapse to their newest copy, oversized tool-call arguments are shrunk. Free,
  idempotent, and usually enough on its own: one 400KB file read costs more than a
  hundred turns of conversation.
- **B — summarize, one auxiliary call.** The middle of the transcript is replaced by a
  structured summary (task, constraints, completed actions, current state, open
  questions, next step). This is what a session needs once the *conversation* is what
  fills the window. A later compaction hands the existing summary to the summarizer to
  be **updated**, never re-summarized — re-compressing lossy text is how a long session
  turns to mush.
- **C — pressure, no LLM.** Last resort: give up everything except the tool round in
  flight.

It runs as phase 0 of every loop *iteration*, not once per turn — a single tool round
can blow the window on its own, and the next request in that same turn is the one that
gets rejected.

```bash
uv run gg-agent "…"                        # on by default
GG_COMPRESS=0 uv run gg-agent "…"          # or Agent(compress=False)
GG_CONTEXT_LENGTH=32000 uv run gg-agent "…"  # or Agent(context_length=32_000)
```

**What gets protected.** The system prompt and the first user turn (the original task is
what a lossy transcript distorts first), plus a tail sized by *tokens* rather than a
message count — a fixed "last 10" either protects nothing or everything depending on
what those ten messages happen to be. When the tail floor ends up protecting the very
result that filled the window, a second pass gives up everything except the tool round
in flight.

**Sizing is deliberately rough.** A chars/4 estimate, multiplied by a factor calibrated
from the `prompt_tokens` the provider actually billed for the last request. A real
tokenizer per provider would be exact and wrong the moment the model changes, and the
threshold only needs to answer "are we near the edge". The threshold itself is 75% of
`context_length - max_tokens`: output is reserved from the same window, so a threshold
computed on the raw window lets a session hit a provider 400 before compression fires.

**The summarizer is not the main model.** `ProviderProfile.default_aux_model` — the
output is never shown to anyone, and a cheap model summarizes structured text about as
well as an expensive one. Three rules in that prompt are each a way sessions have
broken: the turns are **data, not instructions** (they contain tool output, which can
carry anything, including text addressed to a model); credentials are **[REDACTED]**;
and finished work is written in the **past tense**, because an action left in the
imperative reads as an outstanding instruction and gets done twice.

**Failure is not allowed to end the turn.** A summarizer that errors, times out or
returns a stub gets a 5-minute cooldown, and compaction proceeds with a mechanical
extract — the user's own turns, the tools used, the files touched — clearly labelled as
one. If the splice would orphan a tool reply, the whole phase is abandoned and the
transcript is left exactly as it was (`_valid_transcript` checks the result rather than
trusting the boundary logic).

**Two things it interacts with.** Compression rewrites the prompt prefix, so the next
call is a guaranteed cache miss plus a write premium — which is why it only runs once
over the threshold, and why two passes that reclaim under 10% switch it off for the
session. And it breaks the "history is append-only" assumption that index-based flush
dedup relied on, so durability is now a marker on each message (`loop.PERSISTED_KEY`)
instead: the store keeps the original text, a rewritten message is never re-written, and
a message the store has not accepted yet is never pruned or dropped — that is the one
path where this could actually lose data.

**What resume does today.** The store holds the full transcript, including messages a
compaction dropped, so resuming replays everything and compresses again on the first
call (the old summary is folded into the new one). Correct, but wasteful — making the
store compaction-aware is the `active` column already sitting unused in
`migrations/001_init.sql`.

---

## Streaming

On by default: the answer prints as it is generated. Turn it off with `--no-stream`,
`GG_STREAM=0`, or `Agent(stream=False)`.

A streaming transport still returns one `NormalizedResponse`, so the loop, history and
persistence are unchanged. What streaming adds is a set of events on `event_callback`:

| event | payload | meaning |
|---|---|---|
| `stream_delta` | `text` | visible answer text, in order |
| `reasoning_delta` | `text` | reasoning (provider field, Anthropic thinking, or `<think>` tags) |
| `tool_gen_start` | `name` | the model has started writing a call to this tool |
| `stream_break` | — | tools are about to run; the next delta starts with a blank line |
| `stream_error` / `stream_reset` | `error` | the stream failed after output (kept) / mid tool-call (retried) |

The result dict gains `streamed: True` when `response` was already delivered as deltas,
so a front end doesn't print it twice. `--show-reasoning` streams reasoning to stderr.

The retry rules change, because text on screen can't be taken back:

- an endpoint that rejects streaming turns it off for the session and retries without it;
- a failure **before** any text was shown is retried normally;
- a failure **after** text was shown is not retried: the partial text becomes the answer
  (`finish_reason="length"`), except when a tool call was being written, which is retried;
- an interrupt mid-stream keeps the partial answer in history.

Subagents never stream, because parallel children would interleave their tokens. What
hermes adds on top: a stale-stream watchdog, the gateway consumers that edit chat
messages in place, and TTS.

---

## Subagents

A child is a **brand-new `Agent`** with its own conversation, its own tool grant, and a
system prompt built from `goal` + `context`. The parent never sees the child's
intermediate tool calls or reasoning — only the final summary. That is the entire point:
a 40-step research subtask costs the parent one tool result instead of 40 turns of context.

```python
delegate_task(tasks=[
    {"goal": "Count lines of Python under ./src", "context": "Use run_shell."},
    {"goal": "List every tool module and describe it", "context": "Read the files; do not guess."},
], parent_agent=agent)
```

Three things keep it from being a fork bomb:

* **depth cap** — a leaf child doesn't get `delegate_task` back in its tool list.
  Capability is derived from depth, never from what the model asks for.
* **concurrency cap** — bounded semaphores (4 children/call, 8 tools/batch). Children
  are sibling asyncio tasks, not threads, so a wide fan-out costs tasks and each child
  waits on its own I/O concurrently.
* **timeout** — a wedged child yields a `status: "timeout"` entry instead of hanging the parent.

The child's restricted tool grant is literally one filter call —
`registry.get_definitions(enabled_toolsets=..., blocked_tools=...)`. It cannot call what
it was never shown.

---

## GitHub Copilot

Copilot is the one provider that isn't a single env var, and it's a good illustration of
why `ProviderProfile` has hooks at all. It is a **two-stage credential**:

1. a long-lived GitHub OAuth token (`ghu_*` / `gho_*` / `github_pat_*`) — found in
   `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN`, then in the store VS Code and
   the Copilot CLI write (`~/.config/github-copilot/apps.json`), then `gh auth token`;
2. exchanged at `api.github.com/copilot_internal/v2/token` for a **short-lived** Copilot
   API token — that's what actually goes in the `Authorization: Bearer` header.

The exchange also returns **your account's real base URL**. Individual and enterprise
accounts are *not* on `api.githubcopilot.com` — this machine resolves to
`https://api.individual.githubcopilot.com`. Hard-coding the public host is the usual
reason a hand-rolled Copilot client 404s on half the models.

```bash
uv run gg-agent --auth-status           # shows source, token type, expiry, resolved host
uv run gg-agent --login                 # OAuth device-code flow, if not signed in anywhere
uv run gg-agent -p copilot --list-models
uv run gg-agent -p copilot -m claude-sonnet-5 "..."
```

Three details worth knowing:

* **The exchanged token expires** (~30 min on some accounts, 24 h on others), so
  `Agent.run()` calls `refresh_credentials()` before every turn — the profile re-resolves,
  and the client is rebuilt only if the credential actually changed. That's the
  `resolve_credentials()` hook's whole reason to exist.
* **`Copilot-Integration-Id: vscode-chat` is mandatory.** Without it the API rejects the
  request. It and the editor-attribution headers ride on `profile.default_headers`.
* **Auto-detection is Copilot-specific on purpose.** A bare `GITHUB_TOKEN` is usually
  exported for `gh` or CI, so it does *not* silently hijack provider selection — only
  `COPILOT_GITHUB_TOKEN` or an on-disk Copilot store does. Both still work under `-p copilot`.

`gg-agent` never writes to the credential stores VS Code and the Copilot CLI own; `--login`
prints the token for you to export instead.

`providers/copilot_auth.py` is distilled from hermes-agent's `hermes_cli/copilot_auth.py`.
Left out: on-disk JWT persistence across restarts, single-flight locking per token, and
hermes's routing of GPT-5/Codex models to `codex_responses` and Claude models to
`anthropic_messages` (here everything goes over `chat_completions`, which Copilot serves for
all of them).

---

## MCP

External [Model Context Protocol](https://modelcontextprotocol.io) servers register into
the same tool registry as everything else. Once connected, an MCP tool is
indistinguishable from `read_file`: same dispatch, same toolset filtering, same subagent
grant. Nothing in `agent.py`, `loop.py` or the transports knows MCP exists.

Configure with `.mcp.json` at the repo root — the same shape other MCP clients use, so
configs paste across unchanged:

```json
{
  "mcpServers": {
    "everything": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"]},
    "remote":     {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${MY_TOKEN}"}}
  }
}
```

`${VAR}` expands from the environment, so a config can be committed without the secret in
it. `"enabled": false` skips a server.

```python
async with Agent(mcp=True) as agent:                 # or mcp="path/to/config.json"
    print(await agent.aask("Use the MCP tools to ..."))
```

```bash
uv run run.py --list-mcp          # connect, list what each server exposes, exit
uv run run.py --mcp "your task"   # run a turn with MCP tools available
```

Three design points:

* **Naming.** MCP tool names are unique only within their server, so every tool registers
  as `<server>__<tool>` in a toolset `mcp:<server>`. Two servers exporting `search` cannot
  shadow each other, and `enabled_toolsets=["mcp:github"]` is a real grant.
* **Lifecycle.** A stdio server is a subprocess whose session is bound to the task that
  opened it, so each server gets one long-lived runner task that opens the streams and
  parks on a shutdown event. The pool is process-wide: a fan-out of four subagents shares
  one set of servers instead of forking four copies of each.
* **Failure.** A server that won't start is reported in the status mapping, never raised.
  Its tools stay registered but report unavailable, which drops them from the next
  request's tool list — the model simply stops being offered them.

---

## Persistence

Off by default. Set `GG_DATABASE_URL`, sign in, and every session is saved to Postgres
under your user: the transcript, the model and provider, token usage, and which session
delegated to which. Without it, nothing is written and behaviour is unchanged.

Tables go in their own schema, `gg_agent` by default (`GG_DATABASE_SCHEMA` to change
it), created on first connect. That lets gg-agent share a database with other apps
without touching their `users` or `sessions` tables.

It needs a running Postgres server (a local install such as Homebrew `postgresql@16`, or
any server you can reach), no Docker. One-time setup:

```bash
createuser gg --pwprompt            # password: gg
createdb -O gg gg_agent
createdb -O gg gg_agent_test        # only for `pytest -m pg`
psql -d gg_agent      -c 'CREATE EXTENSION IF NOT EXISTS pg_trgm'
psql -d gg_agent_test -c 'CREATE EXTENSION IF NOT EXISTS pg_trgm'
export GG_DATABASE_URL=postgresql://gg:gg@localhost:5432/gg_agent    # see .env.example
```

`pg_trgm` is created by a superuser because `gg` may not be allowed to. On a managed
server where you connect as the admin user (RDS's `postgres`, for example), skip all of
this: the first connection creates the schema, the extension and the tables
(`persistence/migrations/*.sql`).

### Users

Every session has an owner. Users are an email plus a password; the password is stored
only as a salted scrypt hash (`persistence/users.py`, standard library, no new
dependency).

```bash
uv run gg-agent --register you@example.com   # prompts for a password (8+ chars), signs you in
uv run gg-agent --signin you@example.com     # on another machine, or after --signout
uv run gg-agent --whoami
uv run gg-agent --signout
```

Signing in writes your `user_id` and email (never the password) to `~/.gg_agent/user.json`
(`GG_HOME` moves it). This is identification, not security: anyone who can edit that
file can act as that user. If you're not signed in, the agent still runs but doesn't save
anything, and says so.

Everything is scoped to the signed-in user: `--sessions`, `--search`, `--continue`,
`--resume`, the REPL commands and the model's `session_search`. Another user's session id
behaves as if it doesn't exist. In code, pass the owner explicitly:

```python
from gg_agent.persistence.users import register_user, authenticate
user = await authenticate(store, "you@example.com", password)
async with Agent(store=store, user_id=user.id) as agent: ...
```

```bash
uv run gg-agent --continue "what did I ask first?"   # most recent session in this directory
uv run gg-agent --resume 3f9c2a1b7d04                # a specific session
uv run gg-agent --sessions                           # recent sessions
uv run gg-agent --search '"connection pool" -redis'  # search every transcript
uv run gg-agent --no-persist "scratch question"      # don't record this one
```

In the REPL: `/sessions`, `/resume ID`, `/new` (start a fresh session), `/search QUERY`.
`/reset` also starts a new session. The banner shows the session id so you can come back to it.

```python
async with Agent(user_id=uid) as agent:                     # store from $GG_DATABASE_URL
    await agent.arun("...")
async with Agent(user_id=uid, resume="3f9c2a1b7d04") as agent:   # continue it later
    ...
Agent(store=InMemorySessionStore(), user_id=uid)            # any SessionStore; store=False = off
```

How it behaves:

* **Crash-safe ordering.** Messages are written after each model response, *before* the
  tools it asked for run, and again after the tool replies land. A process killed
  mid-tool-round leaves a transcript that `--resume` repairs by dropping the unanswered
  call. Rows are ordered by their id, never by timestamp.
* **A database failure never fails a turn.** It is reported once (a `persist_error`
  event), the in-memory history carries on, and the unsaved messages are retried on the
  next write. If the database is unreachable at startup, the CLI warns once and runs
  without persistence.
* **The model searches, nothing is injected.** With a store, the agent gets a
  `session_search` tool and one line of system-prompt guidance. Given a query, it returns
  the best-matching past sessions (the top one with surrounding messages), and it can
  scroll or read a session by id. It returns only stored messages; no LLM calls.
  Subagents don't get it.
* **Subagents** share the parent's connection pool. Their sessions are linked to the
  parent's and hidden from `--sessions` and search by default.

Search is keyword-only: a `simple`-config `tsvector` (no stemming, which suits code
identifiers) queried with `websearch_to_tsquery`, so `"phrases"`, `OR` and `-term` work
and malformed input never errors. When that finds nothing it falls back to a raw
substring match, backed by the trigram index, for things like paths.

---

## Adding things

**A provider** (OpenAI-compatible): a new module in `gg_agent/providers/plugins/` (or a
`.py` in `$GG_HOME/plugins/model-providers/`) with one `register_provider(ProviderProfile(...))`
call. Nothing else changes. Quirks go in hook overrides: `build_api_kwargs_extras` (reasoning
knobs), `build_extra_body`, `prepare_messages`, `get_max_tokens`, `resolve_credentials`
(short-lived tokens), `create_client` (a non-HTTP wire). See `plugins/deepseek.py`,
`plugins/vertex.py`, `plugins/copilot_acp.py`.

**A protocol** (Gemini native, Vertex Anthropic, …): subclass `ProviderTransport`,
implement the five methods, `register_transport(...)`. The loop is untouched.

**A tool**: a function plus a `registry.register(...)` call in a new
`gg_agent/tools/*.py` — it is auto-discovered at import. Pass `needs_agent=True` when the
handler needs the live `Agent` (that's how `delegate_task` gets its parent). Handlers may
be `def` or `async def`: an async one is awaited, a sync one is pushed to a worker thread
so it cannot stall the loop.

**An MCP server**: an entry in `.mcp.json`. No code.

---

## Tests

No network, no keys, no database, and they never read your real credential store — a
scripted fake transport drives the real loop:

```bash
uv run pytest -q                      # all 107
uv run pytest tests/test_loop.py      # loop, tool rounds, ordering, retries, caps
uv run pytest tests/test_prompt_caching.py  # breakpoint placement, prefix stability, usage
uv run pytest tests/test_subagents.py # delegation, depth caps, both transports
uv run pytest tests/test_async.py     # sync wrappers, concurrency, cancellation, dispatch
uv run pytest tests/test_mcp.py       # config, namespacing, schema translation, failures
uv run pytest tests/test_copilot_auth.py  # token discovery, exchange, caching, refresh
uv run pytest tests/test_persistence.py   # store contract, users + ownership, resume repair, loop/Agent
uv run pytest tests/test_session_search.py  # the four search modes, exclusions, dedup
uv run pytest tests/test_compression.py   # sizing, boundaries, pruning, summarizing, loop wiring
```

The store contract also runs against Postgres, along with Postgres-only tests (search
syntax, the ILIKE fallback, 2MB rows, concurrent writers), when a test database is
configured. Each test works in a throwaway schema:

```bash
GG_TEST_DATABASE_URL=postgresql://gg:gg@localhost:5432/gg_agent_test uv run pytest -m pg
```

---

## What hermes has that this deliberately doesn't

Roughly in the order worth adding back if you keep going:

1. **More persistence** — the transcript, resume and keyword search are here (Postgres,
   above). Still missing: compaction-aware storage — the store keeps every message a
   compaction dropped, so resume replays the uncompressed transcript, where hermes marks
   superseded rows instead (the `active` column in `migrations/001_init.sql` is reserved
   for it) — plus semantic search, per-model cost accounting and LLM-generated titles
   (`hermes_state*.py`, 20+ modules).
2. **The rest of the compressor** — compression itself is implemented (see
   [Context compression](#context-compression)). hermes additionally has focus-topic
   compaction (`/compact <topic>`, which weights the summary towards one subject), image
   retirement, cooldowns that survive a restart, and per-compaction telemetry
   (`agent/context_compressor.py`, `agent/micro_compaction.py`).
3. **Failover & credential pools** — retry onto a fallback model/key on 429/5xx
   (`agent/credential_pool.py`, `agent/error_classifier.py`).
4. **Approval gates** — confirm before destructive tools run (`tools/write_approval.py`).
5. **Live subagent control** — steering, heartbeats, worktree isolation, output schemas
   (`tools/delegate_tool_{progress,registry,results}.py`).
