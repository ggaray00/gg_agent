# scripts/ — run the agent by pressing Run

Every file here is a plain Python script with a small **edit me** block at the
top. No arguments, no flags: open one, change the constants, run it.

Each script does its own `sys.path` setup and loads a `.env` from the repo root,
so nothing needs to be installed and no keys need to be exported first.

| script | what it shows | costs tokens? |
|---|---|---|
| `00_check_setup.py` | providers, credentials, loaded tools | no — start here |
| `01_hello.py` | one question, one answer | yes, tiny |
| `02_tools.py` | the loop calling tools, step by step | yes |
| `03_conversation.py` | multi-turn history | yes |
| `04_subagents.py` | delegation, model-driven and explicit | yes, several turns |
| `05_providers.py` | the same prompt across providers | yes, one turn each |
| `06_playground.py` | every Agent knob, documented — your scratchpad | yes |
| `07_chat.py` | interactive chat, settings in the file | yes |
| `08_mcp.py` | MCP servers as ordinary tools | yes |
| `09_async.py` | the async API; sequential vs concurrent, timed | yes |

`_bootstrap.py` is shared scaffolding (path setup, `.env` loader, event
renderer). It is not part of `gg_agent`.

## Credentials

Put a key in `.env` at the repo root (already gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
# or OPENAI_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY
# GG_PROVIDER=anthropic   # pins the default for every script
```

A real environment variable always wins over `.env`. For a local model with no
key at all, set `PROVIDER = "ollama"` at the top of a script.

## Running

From an editor: open the file, press Run. The working directory in the run
config does not matter — `_bootstrap` chdirs to the repo root on import.

From a terminal, if you prefer:

```bash
python scripts/02_tools.py
```

## Working directory

The file and shell tools resolve relative paths against the **process** working
directory. `Agent(cwd=...)` does *not* redirect them — it only changes what the
system prompt says the working directory is.

PyCharm defaults a run config's working directory to the script's own folder,
so a script in `scripts/` would otherwise send the agent looking for
`gg_agent/` inside `scripts/`. `_bootstrap` chdirs to the repo root to remove
that whole class of confusion. Two ways to override:

- `GG_SCRIPTS_NO_CHDIR=1` — stay wherever you launched from.
- `SCRATCH_DIR` in `06_playground.py` — chdir somewhere disposable, which is a
  real sandbox for `write_file` and `run_shell` because both follow the process
  cwd.

To set it in PyCharm anyway: **Run → Edit Configurations → Working directory**,
set it to the repo root (`$ProjectFileDir$`).

## Sync or async

The core is async all the way down — the loop, the transports, the tools, the
MCP sessions. Both APIs are real:

```python
result = agent.run(prompt)            # sync wrapper, for scripts and tests
result = await agent.arun(prompt)     # native
```

`run` / `ask` / `close` drive a background event loop, so scripts `01`–`07` stay
plain synchronous code. `08` and `09` are `asyncio.run(main())` because they need
the async form. Calling a sync wrapper from inside a running loop raises rather
than deadlocking — use `arun` / `aask` / `aclose` there.

Concurrency is what the async core buys: `09_async.py` runs three agents at once
and measures it (~3× on three questions).

## MCP

`08_mcp.py` reads `.mcp.json` at the repo root — the same
`{"mcpServers": {...}}` shape other MCP clients use, so configs paste across:

```json
{"mcpServers": {"everything": {
   "command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"]}}}
```

Each server becomes a toolset named `mcp:<server>` and each tool registers as
`<server>__<tool>`, so servers can't shadow each other and
`enabled_toolsets=["mcp:github"]` is a real grant. Turn it on per agent:

```python
Agent(mcp=True)              # load .mcp.json
Agent(mcp="path/to.json")    # a specific config
```

`"enabled": false` on a server skips it. A server that fails to start is
reported in the status mapping and its tools drop out of the tool list — it
never stops the agent from running. `python run.py --list-mcp` shows what
each configured server exposes without calling a model.

## Notes

- `02` and `06` default to `VERBOSE = True`, which prints tool results. That is
  the first thing to turn on when a run does something unexpected.
- These scripts run real tools against the real repo. Use `SCRATCH_DIR` in
  `06_playground.py` before asking for file writes.
- `BLOCKED_TOOLS = {"write_file", "run_shell"}` gives a read-only run.
