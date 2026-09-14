#!/usr/bin/env python3
"""An interactive chat you start by running the file — no arguments to remember.

Same idea as `python run.py` with no prompt, but the settings live at the top of
this file instead of on a command line, and you can edit them between runs.

    /reset    clear history      /history  message count
    /tools    list the tool grant   /exit   quit (Ctrl-D works too)
"""

from _bootstrap import banner, footer, make_renderer, preflight

# ── edit me ──────────────────────────────────────────────────────────────
PROVIDER = None
MODEL = None
VERBOSE = False
TOOLSETS = None
EXTRA_INSTRUCTIONS = ""
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not preflight():
        return 2

    from gg_agent import Agent

    with Agent(provider=PROVIDER, model=MODEL, enabled_toolsets=TOOLSETS,
               extra_instructions=EXTRA_INSTRUCTIONS,
               event_callback=make_renderer(VERBOSE)) as agent:
        banner(agent, "chat")
        print("Type a task. /reset · /history · /tools · /exit")

        while True:
            try:
                line = input("\n› ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                return 0
            if line == "/reset":
                agent.reset()
                print("history cleared")
                continue
            if line == "/history":
                print(f"{len(agent.history)} messages")
                continue
            if line == "/tools":
                for definition in agent.tool_definitions():
                    name = definition.get("name") or definition.get("function", {}).get("name")
                    print(f" • {name}")
                continue
            try:
                result = agent.run(line)
            except KeyboardInterrupt:
                # Cooperative stop: the loop checks this between iterations.
                agent.interrupt()
                print("\ninterrupted")
                agent.clear_interrupt()
                continue
            print(f"\n{result['response']}")
            footer(result)


if __name__ == "__main__":
    raise SystemExit(main())
