"""Terminal front-end: one-shot or interactive REPL.

Mirrors hermes-agent: cli.py (217k lines there — this is the 1% that runs a turn).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import sys
from pathlib import Path

from .agent import Agent, resolve_provider
from .aio import run_sync
from .home import ensure_gg_home, get_gg_home, get_working_dir
from .providers import list_providers
from .tools.registry import discover_builtin_tools, registry

# Event rendering. hermes-agent has a whole display layer (spinners, boxes,
# markdown tables); this is the same event stream printed plainly. Streamed answer
# text goes to stdout as it arrives; everything else is status on stderr.
_ICONS = {"api_call": "🤖", "tool_start": "🔧", "delegate_start": "🔀", "api_retry": "⚠️"}
_DIM, _RESET = ("\033[2m", "\033[0m") if sys.stderr.isatty() else ("", "")


def make_renderer(verbose: bool, show_reasoning: bool = False):
    # Whether stdout sits at the start of a line, so status lines on stderr never
    # land in the middle of a streamed sentence.
    state = {"at_line_start": True, "reasoning_open": False}

    def end_line() -> None:
        if state["reasoning_open"]:
            print(_RESET, file=sys.stderr, flush=True)
            state["reasoning_open"] = False
        if not state["at_line_start"]:
            print(flush=True)
            state["at_line_start"] = True

    def render(kind: str, payload: dict) -> None:
        indent = "  " * payload.get("depth", 0)
        if kind == "stream_delta":
            if state["reasoning_open"]:
                print(_RESET, file=sys.stderr, flush=True)
                state["reasoning_open"] = False
            text = payload["text"]
            print(text, end="", flush=True)
            state["at_line_start"] = text.endswith("\n")
            return
        if kind == "reasoning_delta":
            if show_reasoning or verbose:
                if not state["reasoning_open"]:
                    end_line()
                    print(f"{indent}💭 {_DIM}", end="", file=sys.stderr)
                    state["reasoning_open"] = True
                print(payload["text"], end="", file=sys.stderr, flush=True)
            return
        if kind == "stream_break":
            end_line()
            return
        if kind in {"api_call", "tool_gen_start", "tool_start", "tool_end", "delegate_start",
                    "api_retry", "stream_error", "stream_reset", "mcp_server", "persist_error",
                    "context_compressed", "context_full", "summarizing", "summary_failed",
                    "compression_disabled"}:
            end_line()
        if kind == "api_call":
            print(f"{indent}{_ICONS[kind]} call #{payload['iteration']} → {payload['model']}", file=sys.stderr)
        elif kind == "tool_gen_start":
            print(f"{indent}⚡ preparing {payload['name']}…", file=sys.stderr)
        elif kind == "tool_start":
            args = payload.get("args") or {}
            preview = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in list(args.items())[:3])
            print(f"{indent}🔧 {payload['name']}({preview})", file=sys.stderr)
        elif kind == "tool_end" and verbose:
            print(f"{indent}   ↳ {str(payload.get('result'))[:200]}", file=sys.stderr)
        elif kind == "delegate_start":
            print(f"{indent}🔀 delegating {payload['count']} task(s):", file=sys.stderr)
            for goal in payload.get("goals", []):
                print(f"{indent}   • {goal}", file=sys.stderr)
        elif kind == "api_retry":
            print(f"{indent}⚠️  {payload['error']} — retrying in {payload['delay']:.1f}s", file=sys.stderr)
        elif kind == "stream_error":
            print(f"{indent}⚠️  stream failed after partial output, keeping what arrived: {payload['error']}",
                  file=sys.stderr)
        elif kind == "stream_reset":
            print(f"{indent}⚠️  connection dropped mid tool-call; reconnecting…", file=sys.stderr)
        elif kind == "summarizing":
            print(f"{indent}🗜️  summarizing {payload['messages']} older messages "
                  f"({payload['tokens']:,} tokens)…", file=sys.stderr)
        elif kind == "summary_failed":
            print(f"{indent}⚠️  summary failed ({payload['error']}) — keeping a mechanical "
                  "extract instead", file=sys.stderr)
        elif kind == "compression_disabled":
            print(f"{indent}⚠️  compression is not freeing enough space ({payload['tokens']:,} "
                  "tokens); turning it off for this session — /new starts fresh", file=sys.stderr)
        elif kind == "context_compressed":
            print(f"{indent}🗜️  context {payload['before']:,} → {payload['after']:,} tokens "
                  f"(threshold {payload['threshold']:,})", file=sys.stderr)
        elif kind == "context_full":
            print(f"{indent}⚠️  context at {payload['tokens']:,} tokens with nothing left to prune "
                  f"(threshold {payload['threshold']:,}) — start a new session with /new",
                  file=sys.stderr)
        elif kind == "mcp_server":
            print(f"{indent}🔌 mcp/{payload['server']}: {payload['status']}", file=sys.stderr)
        elif kind == "persist_error":
            if payload.get("stage") == "open":
                print(f"{indent}⚠️  database unreachable, running without persistence: {payload['error']}",
                      file=sys.stderr)
            elif payload.get("stage") == "owner":
                print(f"{indent}⚠️  not saving sessions: {payload['error']} — sign in again with "
                      "--signin EMAIL", file=sys.stderr)
            else:
                print(f"{indent}⚠️  persistence ({payload.get('stage')}): {payload['error']}", file=sys.stderr)

    render.end_line = end_line
    return render


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gg-agent", description="A minimal agentic loop.")
    p.add_argument("prompt", nargs="*", help="Task to run. Omit for an interactive REPL.")
    p.add_argument("-p", "--provider", help="openai | anthropic | openrouter | groq | deepseek | ollama")
    p.add_argument("-m", "--model", help="Model id (defaults to the provider's default).")
    p.add_argument("-t", "--toolsets", help="Comma-separated toolsets to enable (default: all).")
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=2, help="Subagent nesting cap.")
    p.add_argument("--no-delegation", action="store_true", help="Block the delegate_task tool.")
    p.add_argument("-v", "--verbose", action="store_true", help="Show tool results and reasoning too.")
    p.add_argument("--no-stream", action="store_true",
                   help="Print the answer when it is complete instead of as it arrives (or GG_STREAM=0).")
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="Don't mark the prompt for caching (Anthropic; other providers cache anyway).")
    p.add_argument("--show-reasoning", action="store_true",
                   help="Stream the model's reasoning to stderr, when it exposes any.")
    p.add_argument("--list-providers", action="store_true", help="Show providers and credential status.")
    p.add_argument("--list-tools", action="store_true")
    p.add_argument("--list-models", action="store_true",
                   help="Ask the provider which models this account can actually use.")
    p.add_argument("--login", nargs="?", const="copilot", metavar="PROVIDER",
                   help="Run the OAuth device-code flow (currently: copilot).")
    p.add_argument("--auth-status", action="store_true",
                   help="Show where each provider's credential is coming from.")
    p.add_argument("--mcp", nargs="?", const=True, metavar="CONFIG",
                   help="Connect MCP servers from .mcp.json (or the given config file).")
    p.add_argument("--list-mcp", action="store_true",
                   help="Connect the configured MCP servers, list their tools, and exit.")
    # Persistence (on when GG_DATABASE_URL is set).
    p.add_argument("--resume", metavar="ID", help="Continue a saved session by id.")
    p.add_argument("--continue", dest="continue_", action="store_true",
                   help="Continue the most recent session in this directory.")
    p.add_argument("--sessions", action="store_true", help="List recent sessions and exit.")
    p.add_argument("--search", metavar="QUERY", help="Search past sessions and exit.")
    p.add_argument("--no-persist", action="store_true", help="Don't save this session.")
    # Users: every saved session has an owner.
    p.add_argument("--register", metavar="EMAIL", help="Create a user (prompts for a password) and sign in.")
    p.add_argument("--signin", metavar="EMAIL", help="Sign in as an existing user.")
    p.add_argument("--signout", action="store_true", help="Forget the signed-in user on this machine.")
    p.add_argument("--whoami", action="store_true", help="Show the signed-in user.")
    return p


# ── Signed-in user ──────────────────────────────────────────────────────────
# Just "who am I" for this machine: the user_id and email, no password and no
# token. The password is checked against the database at --signin / --register.

def _account_file() -> Path:
    return get_gg_home() / "user.json"


def _load_account() -> dict | None:
    try:
        account = json.loads(_account_file().read_text())
    except (OSError, ValueError):
        return None
    return account if isinstance(account, dict) and account.get("user_id") else None


def _save_account(user) -> None:
    path = _account_file()
    ensure_gg_home()
    path.write_text(json.dumps({"user_id": user.id, "email": user.email}))
    path.chmod(0o600)


def _read_password(confirm: bool) -> str | None:
    password = getpass.getpass("Password: ")
    if confirm and getpass.getpass("Repeat password: ") != password:
        print("error: passwords don't match", file=sys.stderr)
        return None
    return password


async def _cmd_register(email: str, password: str) -> int:
    from .persistence.users import register_user

    async def do(store):
        user = await register_user(store, email, password)
        _save_account(user)
        print(f"registered and signed in as {user.email} (user_id {user.id})", file=sys.stderr)
    return await _with_default_store(do)


async def _cmd_signin(email: str, password: str) -> int:
    from .persistence.users import authenticate

    async def do(store):
        user = await authenticate(store, email, password)
        if user is None:
            raise ValueError("wrong email or password")
        _save_account(user)
        print(f"signed in as {user.email} (user_id {user.id})", file=sys.stderr)
    return await _with_default_store(do)


def _cmd_account(args) -> int:
    if args.signout:
        _account_file().unlink(missing_ok=True)
        print("signed out", file=sys.stderr)
        return 0
    account = _load_account()                           # --whoami
    print(f"{account['email']} (user_id {account['user_id']})" if account else "not signed in")
    return 0 if account else 1


# ── Sessions ────────────────────────────────────────────────────────────────

def _print_sessions(sessions, current: str | None = None) -> None:
    if not sessions:
        print("no sessions yet", file=sys.stderr)
    for info in sessions:
        when = info.last_activity_at.strftime("%Y-%m-%d %H:%M") if info.last_activity_at else "?"
        mark = "*" if info.id == current else " "
        print(f"{mark} {info.id}  {when}  {info.message_count:>4} msgs  {info.title or '(untitled)'}")


def _print_hits(hits) -> None:
    if not hits:
        print("no matches", file=sys.stderr)
    for hit in hits:
        snippet = " ".join((hit.snippet or "").split())
        print(f"{hit.session_id}  #{hit.message_id:<6} {hit.role:<9} {snippet}")


async def _with_default_store(fn, *, need_account: bool = False) -> int:
    """Open the $GG_DATABASE_URL store, run ``fn(store)``, close it.
    A ValueError from ``fn`` is a user-facing error, printed and returned as 2."""
    from .persistence import get_default_store

    if need_account and _load_account() is None:
        print("error: not signed in — use --signin EMAIL or --register EMAIL.", file=sys.stderr)
        return 2
    store = get_default_store()
    if store is None:
        print("error: persistence is not configured — set GG_DATABASE_URL.", file=sys.stderr)
        return 2
    try:
        await store.open()
    except Exception as exc:
        print(f"error: database unreachable: {exc}", file=sys.stderr)
        return 1
    try:
        await fn(store)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        await store.close()
    return 0


async def _cmd_sessions() -> int:
    async def show(store):
        _print_sessions(await store.list_sessions(limit=20, owner_id=_load_account()["user_id"]))
    return await _with_default_store(show, need_account=True)


async def _cmd_search(query: str) -> int:
    async def show(store):
        _print_hits(await store.search(query, limit=20, owner_id=_load_account()["user_id"]))
    return await _with_default_store(show, need_account=True)


async def _latest_session_id(agent: Agent) -> str | None:
    sessions = await agent.store.list_sessions(limit=1, cwd=agent.cwd, owner_id=agent.user_id)
    return sessions[0].id if sessions else None


def _cmd_login(provider: str) -> int:
    """OAuth device-code login. Only Copilot needs one today."""
    if provider not in {"copilot", "github", "github-copilot"}:
        print(f"error: no OAuth flow for provider {provider!r} (only copilot).", file=sys.stderr)
        return 2
    from .providers.copilot_auth import device_code_login

    token = device_code_login()
    if not token:
        return 1
    # gg-agent deliberately does NOT write to the credential stores that VS Code and
    # the Copilot CLI own — clobbering those would break the user's editor session.
    print("\n  ✓ Authorized. Export this token to use it:\n")
    print(f"    export COPILOT_GITHUB_TOKEN={token}\n")
    print("  (Or add it to your shell profile. If you signed in through VS Code or the")
    print("   Copilot CLI, gg-agent already finds that token on disk — no export needed.)")
    return 0


async def _cmd_list_mcp(config_path) -> int:
    """Connect, print what each server exposes, disconnect."""
    from .tools.mcp_tools import close_mcp_servers, connect_mcp_servers, load_mcp_config

    configs = load_mcp_config(config_path)
    if not configs:
        print("no MCP servers configured (looked for .mcp.json, .claude/mcp.json)", file=sys.stderr)
        return 1
    try:
        status = await connect_mcp_servers(config_path)
        for name, state in status.items():
            print(f"{name:<16} {state}")
        for name in registry.all_names():
            entry = registry.get(name)
            if entry.toolset.startswith("mcp:"):
                print(f"  🔌 {name:<28} {entry.description.splitlines()[0][:70] if entry.description else ''}")
    finally:
        await close_mcp_servers()
    return 0


def _cmd_auth_status() -> int:
    for profile in list_providers():
        try:
            status = profile.credential_status()
        except Exception as exc:
            status = f"error: {exc}"
        mark = "✓" if profile.has_credentials() else " "
        print(f"{mark} {profile.name:<12} {status}")
    print("\n✓ = auto-detected. Others still work when named explicitly with -p.", file=sys.stderr)
    return 0


async def _cmd_list_models(provider: str | None) -> int:
    """Live model catalog. Worth asking for Copilot especially: which models an
    account can see depends on its plan and org policy, not on a static list."""
    try:
        profile = resolve_provider(provider)
        api_key, base_url = profile.resolve_credentials()
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if profile.api_mode != "chat_completions":
        for model in profile.fallback_models or (profile.default_model,):
            print(model)
        print(f"\n({profile.name} has no catalog endpoint; showing the declared list.)", file=sys.stderr)
        return 0
    from .transports import get_transport

    transport = get_transport("chat_completions")
    client = transport.build_client(api_key=api_key, base_url=base_url, profile=profile)
    try:
        models = sorted(m.id for m in (await client.models.list()).data)
    except Exception as exc:
        print(f"error: could not list models for {profile.name}: {exc}", file=sys.stderr)
        return 1
    finally:
        await transport.aclose_client(client)
    for model in models:
        print(model)
    print(f"\n[{len(models)} models · {profile.name} · {base_url}]", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    # Home (config/state) and working dir (what the tools act on) are decided
    # once, here, before anything reads either. GG_CWD is the anchor a tool falls
    # back to if the launch directory disappears mid-run; setdefault so a caller
    # that already pinned one (a wrapper, a test) keeps it.
    ensure_gg_home()
    os.environ.setdefault("GG_CWD", get_working_dir())

    if args.login:
        return _cmd_login(args.login.lower())

    if args.auth_status:
        return _cmd_auth_status()

    if args.list_providers:
        for profile in list_providers():
            state = "configured" if profile.has_credentials() else "-"
            print(f"{profile.name:<12} {profile.api_mode:<20} {state:<12} {profile.default_model}")
        return 0

    if args.list_models:
        return asyncio.run(_cmd_list_models(args.provider))

    if args.list_mcp:
        return asyncio.run(_cmd_list_mcp(args.mcp if args.mcp is not True else None))

    if args.register or args.signin:
        password = _read_password(confirm=bool(args.register))
        if password is None:
            return 2
        if args.register:
            return asyncio.run(_cmd_register(args.register, password))
        return asyncio.run(_cmd_signin(args.signin, password))

    if args.signout or args.whoami:
        return _cmd_account(args)

    if args.sessions:
        return asyncio.run(_cmd_sessions())

    if args.search:
        return asyncio.run(_cmd_search(args.search))

    if args.list_tools:
        discover_builtin_tools()
        for name in registry.all_names():
            entry = registry.get(name)
            print(f"{entry.emoji} {name:<16} [{entry.toolset}] {entry.description.splitlines()[0][:80]}")
        return 0

    account = _load_account()
    wants_persistence = bool(os.getenv("GG_DATABASE_URL", "").strip()) and not args.no_persist
    if wants_persistence and account is None:
        if args.resume or args.continue_:
            print("error: not signed in — use --signin EMAIL first.", file=sys.stderr)
            return 2
        print("⚠️  not signed in, so this session won't be saved "
              "(gg-agent --register EMAIL, or --signin EMAIL)", file=sys.stderr)

    try:
        agent = Agent(
            provider=args.provider,
            model=args.model,
            mcp=args.mcp,
            enabled_toolsets=[t.strip() for t in args.toolsets.split(",")] if args.toolsets else None,
            blocked_tools={"delegate_task"} if args.no_delegation else None,
            max_iterations=args.max_iterations,
            max_depth=args.max_depth,
            event_callback=make_renderer(args.verbose, args.show_reasoning),
            stream=False if args.no_stream else None,
            prompt_caching=not args.no_prompt_cache,
            store=None if wants_persistence and account else False,
            user_id=account["user_id"] if account else None,
            resume=args.resume,
        )
        # Open the store up front: an unreachable DB is reported (and persistence
        # switched off) before the first prompt, and --resume fails fast.
        persisting = run_sync(agent.astart())
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.continue_:
        latest = run_sync(_latest_session_id(agent)) if persisting else None
        if latest:
            agent.resume(latest)
        else:
            print("no previous session to continue here; starting a new one", file=sys.stderr)

    banner = f"[{agent.profile.name}/{agent.model} · {len(agent.tool_definitions())} tools"
    if persisting:
        banner += f" · {account['email']} · session {agent.session_id}"
        if agent.history:
            banner += f" (resumed, {len(agent.history)} messages)"
    print(banner + "]", file=sys.stderr)

    with agent:
        if args.prompt:
            return _run_once(agent, " ".join(args.prompt))
        return _repl(agent)


def _run_once(agent: Agent, prompt: str) -> int:
    result = agent.run(prompt)
    _print_response(agent, result)
    _print_footer(result)
    return 1 if result["failed"] else 0


def _print_response(agent: Agent, result: dict) -> None:
    """Print the answer — unless it was already streamed onto the screen."""
    end_line = getattr(agent.event_callback, "end_line", None)
    if end_line is not None:
        end_line()
    if not result.get("streamed"):
        print(result["response"])


def _print_footer(result: dict) -> None:
    usage = result["usage"]
    print(f"\n[{result['api_calls']} calls · {result['tool_calls']} tools · "
          f"{usage.prompt_tokens}+{usage.completion_tokens} tokens · "
          f"{result['duration_seconds']}s · {result['exit_reason']}]", file=sys.stderr)


def _repl_session_command(agent: Agent, line: str) -> bool:
    """Handle /sessions, /resume, /new, /search. Returns False if it wasn't one."""
    command, _, arg = line.partition(" ")
    arg = arg.strip()
    if command not in {"/sessions", "/resume", "/new", "/search"}:
        return False
    if agent.store is None:
        print("persistence is off (set GG_DATABASE_URL)", file=sys.stderr)
        return True
    try:
        if command == "/sessions":
            _print_sessions(run_sync(agent.store.list_sessions(limit=20, owner_id=agent.user_id)),
                            current=agent.session_id)
        elif command == "/search":
            if not arg:
                print("usage: /search QUERY", file=sys.stderr)
            else:
                _print_hits(run_sync(agent.store.search(arg, limit=20, owner_id=agent.user_id)))
        elif command == "/resume":
            if not arg:
                print("usage: /resume ID", file=sys.stderr)
            else:
                agent.resume(arg)
                print(f"resumed {agent.session_id} ({len(agent.history)} messages)", file=sys.stderr)
        else:
            agent.reset()
            print(f"new session {agent.session_id}", file=sys.stderr)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
    return True


def _repl(agent: Agent) -> int:
    help_text = "Type a task, or /reset, /history, /exit."
    if agent.store is not None:
        help_text = "Type a task, or /new, /sessions, /resume ID, /search QUERY, /history, /exit."
    print(help_text, file=sys.stderr)
    while True:
        try:
            line = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            return 0
        if line == "/reset":
            agent.reset()
            print("history cleared" + (f" · new session {agent.session_id}" if agent.store else ""),
                  file=sys.stderr)
            continue
        if _repl_session_command(agent, line):
            continue
        if line == "/history":
            print(f"{len(agent.history)} messages", file=sys.stderr)
            continue
        try:
            result = agent.run(line)
        except KeyboardInterrupt:
            agent.interrupt()
            print("\ninterrupted", file=sys.stderr)
            agent.clear_interrupt()
            continue
        _print_response(agent, result)
        _print_footer(result)


if __name__ == "__main__":
    raise SystemExit(main())
