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

uv run pytest                            # 26 offline tests, no keys needed
uv run ruff check gg_agent/
```

Credentials, in the order `gg-agent` looks for them:

| Provider | How to authenticate |
|---|---|
| **GitHub Copilot** | already signed in via VS Code / Copilot CLI → **nothing to do**; else `uv run gg-agent --login` |
| anthropic / openai / openrouter / groq / deepseek | `export ANTHROPIC_API_KEY=…` (etc.) |
| ollama | nothing — `uv run gg-agent -p ollama` |

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
| `providers/__init__.py` | registry + 7 built-in profiles | `providers/__init__.py`, `plugins/model-providers/` |
| `providers/copilot_auth.py` | Copilot OAuth: token discovery, exchange, caching, device login | `hermes_cli/copilot_auth.py` |
| `transports/types.py` | `ToolCall` / `Usage` / `NormalizedResponse` — the only types the loop sees | `agent/transports/types.py` |
| `transports/base.py` | `ProviderTransport` ABC: convert → build → call → normalize | `agent/transports/base.py` |
| `transports/chat_completions.py` | OpenAI + every OpenAI-compatible endpoint | `agent/transports/chat_completions.py` |
| `transports/anthropic.py` | Messages API: system extraction, `tool_use`/`tool_result` blocks | `agent/transports/anthropic.py` |
| `tools/registry.py` | `ToolEntry`, registration, toolset filtering, safe dispatch | `tools/registry.py` + `model_tools.handle_function_call` |
| `tools/shell_tool.py` | `run_shell` | `tools/terminal_tool*.py` |
| `tools/file_tools.py` | `read_file` / `write_file` / `list_dir` | `tools/file_tools.py`, `file_operations_*.py` |
| `tools/delegate_tool.py` | `delegate_task`: parallel subagents, depth cap, timeouts | `tools/delegate_tool*.py` (7 modules) |
| `tools/mcp_tools.py` | MCP servers registered as ordinary tools: config, lifecycle, translation | `plugins/mcp/*` |
| `aio.py` | the one sync↔async boundary: a persistent background loop | — |
| `loop.py` | `run_conversation` + `LoopState` + phases + parallel tool execution | `agent/conversation_loop.py`, `agent/turn_*.py`, `agent/tool_executor.py` |
| `agent.py` | `Agent`: provider resolution, client, tool grant, interrupts, turn facade | `run_agent.AIAgent`, `agent/agent_init.py`, `agent/client_lifecycle.py` |
| `prompts.py` | main + child system prompts | `agent/prompt_builder.py`, `tools/delegate_tool_progress.py` |
| `cli.py` | one-shot / REPL front-end, event rendering | `cli.py` |

**The `LoopState` + phase-function shape in `loop.py` is not decoration.** It is exactly
how hermes scales that loop: each phase (`assemble_request`, `perform_api_call`,
`run_tool_round`, …) reads the state fields it needs and rebinds the ones it owns, so a
phase can move into its own module without changing a signature. hermes has ~20 such
`agent/turn_*.py` modules threading one dataclass.

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

## Adding things

**A provider** (OpenAI-compatible): one `register_provider(ProviderProfile(...))` call in
`providers/__init__.py`. Nothing else changes.

**A protocol** (Gemini native, Bedrock, Responses API): subclass `ProviderTransport`,
implement the five methods, `register_transport(...)`. The loop is untouched.

**A tool**: a function plus a `registry.register(...)` call in a new
`gg_agent/tools/*.py` — it is auto-discovered at import. Pass `needs_agent=True` when the
handler needs the live `Agent` (that's how `delegate_task` gets its parent). Handlers may
be `def` or `async def`: an async one is awaited, a sync one is pushed to a worker thread
so it cannot stall the loop.

**An MCP server**: an entry in `.mcp.json`. No code.

---

## Tests

No network, no keys, and they never read your real credential store — a scripted fake
transport drives the real loop:

```bash
uv run pytest -q                      # all 38
uv run pytest tests/test_loop.py      # loop, tool rounds, ordering, retries, caps
uv run pytest tests/test_subagents.py # delegation, depth caps, both transports
uv run pytest tests/test_async.py     # sync wrappers, concurrency, cancellation, dispatch
uv run pytest tests/test_mcp.py       # config, namespacing, schema translation, failures
uv run pytest tests/test_copilot_auth.py  # token discovery, exchange, caching, refresh
```

---

## What hermes has that this deliberately doesn't

Roughly in the order worth adding back if you keep going:

1. **Streaming** — token deltas to a callback (`agent/stream_delivery.py`).
2. **Context compression** — summarize old turns before the window overflows
   (`agent/context_compressor.py`, `trajectory_compressor.py`). Everything else stays
   usable without this; long sessions do not.
3. **Persistence** — SQLite transcript, resume, search (`hermes_state*.py`, 20+ modules).
4. **Failover & credential pools** — retry onto a fallback model/key on 429/5xx
   (`agent/credential_pool.py`, `agent/error_classifier.py`).
5. **Approval gates** — confirm before destructive tools run (`tools/write_approval.py`).
6. **Prompt caching** — `cache_control` breakpoints; large cost win on long sessions
   (`agent/prompt_caching.py`).
7. **Live subagent control** — steering, heartbeats, worktree isolation, output schemas
   (`tools/delegate_tool_{progress,registry,results}.py`).
