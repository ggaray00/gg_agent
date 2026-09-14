"""Shared setup for everything in scripts/ — import this first.

Every script here is meant to be run by pressing Run in an editor, so this
module does the four things a CLI would otherwise do for you:

  1. put the repo root on sys.path (no `pip install -e .` needed),
  2. chdir to the repo root, so tool paths mean the same thing however the
     script was launched (see the note below),
  3. load a `.env` at the repo root so keys don't have to be exported,
  4. hand out a printable event renderer and a run footer.

On (2): the file and shell tools resolve relative paths against the PROCESS
working directory, not against ``agent.cwd`` — ``Agent(cwd=...)`` only changes
what the system prompt claims. PyCharm defaults the working directory to the
script's own folder, so without this chdir the agent looks for ``gg_agent/``
inside ``scripts/`` and gets "not a directory". Set GG_SCRIPTS_NO_CHDIR=1 to
opt out and run against wherever you launched from.

Nothing here is part of gg_agent itself; it is scaffolding for testing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Anchor the process cwd; the tools resolve relative paths against it.
if not os.getenv("GG_SCRIPTS_NO_CHDIR"):
    os.chdir(ROOT)


def load_dotenv(path: Path | None = None) -> list[str]:
    """Minimal KEY=VALUE reader — no dependency, and never clobbers a real env var.

    Supports `export KEY=value`, `#` comments, and surrounding quotes. Returns the
    names it set so a script can say where its credentials came from.
    """
    env_file = path or (ROOT / ".env")
    if not env_file.is_file():
        return []
    loaded = []
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:      # a real env var always wins
            os.environ[key] = value
            loaded.append(key)
    return loaded


def make_renderer(verbose: bool = False):
    """Print the agent's event stream as it happens. Pass as ``event_callback=``.

    ``verbose=True`` also prints tool results, which is the single most useful
    switch when a run does something you didn't expect.
    """

    def render(kind: str, payload: dict) -> None:
        indent = "  " * payload.get("depth", 0)
        if kind == "api_call":
            print(f"{indent}🤖 call #{payload['iteration']} → {payload['model']}")
        elif kind == "tool_start":
            args = payload.get("args") or {}
            preview = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in list(args.items())[:3])
            print(f"{indent}🔧 {payload['name']}({preview})")
        elif kind == "tool_end" and verbose:
            print(f"{indent}   ↳ {str(payload.get('result'))[:300]}")
        elif kind == "delegate_start":
            print(f"{indent}🔀 delegating {payload['count']} task(s):")
            for goal in payload.get("goals", []):
                print(f"{indent}   • {goal}")
        elif kind == "api_retry":
            print(f"{indent}⚠️  {payload['error']} — retrying in {payload['delay']:.1f}s")

    return render


def banner(agent, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    # os.getcwd(), not agent.cwd: the tools resolve paths against the process.
    print(f"[{agent.profile.name}/{agent.model} · {len(agent.tool_definitions())} tools · "
          f"cwd={os.getcwd()}]\n")


def footer(result: dict) -> None:
    """The same run stats the CLI prints, so scripts stay comparable to `run.py`."""
    usage = result["usage"]
    print(f"\n[{result['api_calls']} calls · {result['tool_calls']} tools · "
          f"{usage.prompt_tokens}+{usage.completion_tokens} tokens · "
          f"{result['duration_seconds']}s · {result['exit_reason']}]")


def preflight() -> bool:
    """Check a provider is reachable before a script burns a turn on a stack trace."""
    load_dotenv()
    from gg_agent.agent import resolve_provider

    try:
        profile = resolve_provider(os.getenv("GG_PROVIDER") or None)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}\n")
        print("Fix: put a key in a .env file at the repo root, e.g.")
        print("    ANTHROPIC_API_KEY=sk-ant-...")
        print("or set PROVIDER at the top of this script to 'ollama' for a local model.")
        return False
    print(f"provider: {profile.name} ({profile.display_name}) · default model {profile.default_model}")
    return True
