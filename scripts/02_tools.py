#!/usr/bin/env python3
"""Watch the tool loop work.

The renderer prints every API call and every tool invocation as it happens, so
you can see the loop iterate instead of only seeing the final answer.

Set VERBOSE = True to print tool RESULTS too — that is the switch you want when
the agent does something surprising.
"""

from _bootstrap import answer, banner, footer, make_renderer, preflight

# ── edit me ──────────────────────────────────────────────────────────────
PROVIDER = None
MODEL = None
VERBOSE = True           # True also prints each tool's result
TOOLSETS = None          # None = all. Or ["files"], ["shell"], ["files", "shell"]
PROMPT = (
    "Look at this repository and tell me: how many Python files are in gg_agent/, "
    "and what does gg_agent/loop.py do? Be concise."
)
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not preflight():
        return 2

    from gg_agent import Agent

    with Agent(
        provider=PROVIDER,
        model=MODEL,
        enabled_toolsets=TOOLSETS,
        event_callback=make_renderer(VERBOSE),
        max_iterations=20,
    ) as agent:
        banner(agent, "tool loop")
        result = agent.run(PROMPT)
        answer(result)
        footer(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
