"""Terminal front-end: one-shot or interactive REPL.

Mirrors hermes-agent: cli.py (217k lines there — this is the 1% that runs a turn).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .agent import Agent, resolve_provider
from .providers import list_providers
from .tools.registry import discover_builtin_tools, registry

# Event rendering. hermes-agent has a whole display layer (spinners, streaming,
# per-tool progress); this is the same event stream printed plainly.
_ICONS = {"api_call": "🤖", "tool_start": "🔧", "delegate_start": "🔀", "api_retry": "⚠️"}


def make_renderer(verbose: bool):
    def render(kind: str, payload: dict) -> None:
        indent = "  " * payload.get("depth", 0)
        if kind == "api_call":
            print(f"{indent}{_ICONS[kind]} call #{payload['iteration']} → {payload['model']}", file=sys.stderr)
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
        elif kind == "mcp_server":
            print(f"{indent}🔌 mcp/{payload['server']}: {payload['status']}", file=sys.stderr)
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
    p.add_argument("-v", "--verbose", action="store_true", help="Show tool results too.")
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
    return p


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

    if args.list_tools:
        discover_builtin_tools()
        for name in registry.all_names():
            entry = registry.get(name)
            print(f"{entry.emoji} {name:<16} [{entry.toolset}] {entry.description.splitlines()[0][:80]}")
        return 0

    try:
        agent = Agent(
            provider=args.provider,
            model=args.model,
            mcp=args.mcp,
            enabled_toolsets=[t.strip() for t in args.toolsets.split(",")] if args.toolsets else None,
            blocked_tools={"delegate_task"} if args.no_delegation else None,
            max_iterations=args.max_iterations,
            max_depth=args.max_depth,
            event_callback=make_renderer(args.verbose),
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"[{agent.profile.name}/{agent.model} · {len(agent.tool_definitions())} tools]", file=sys.stderr)

    with agent:
        if args.prompt:
            return _run_once(agent, " ".join(args.prompt))
        return _repl(agent)


def _run_once(agent: Agent, prompt: str) -> int:
    result = agent.run(prompt)
    print(result["response"])
    _print_footer(result)
    return 1 if result["failed"] else 0


def _print_footer(result: dict) -> None:
    usage = result["usage"]
    print(f"\n[{result['api_calls']} calls · {result['tool_calls']} tools · "
          f"{usage.prompt_tokens}+{usage.completion_tokens} tokens · "
          f"{result['duration_seconds']}s · {result['exit_reason']}]", file=sys.stderr)


def _repl(agent: Agent) -> int:
    print("Type a task, or /reset, /history, /exit.", file=sys.stderr)
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
            print("history cleared", file=sys.stderr)
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
        print(result["response"])
        _print_footer(result)


if __name__ == "__main__":
    raise SystemExit(main())
