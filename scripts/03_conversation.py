#!/usr/bin/env python3
"""Multi-turn: history carries between turns.

Each entry in TURNS is a separate call to the model, but the agent keeps the
transcript, so turn 2 can refer to what happened in turn 1. Compare with
KEEP_HISTORY = False, where every turn starts cold.
"""

from _bootstrap import answer, banner, footer, make_renderer, preflight

# ── edit me ──────────────────────────────────────────────────────────────
PROVIDER = None
MODEL = None
KEEP_HISTORY = True      # flip to False to see each turn lose the thread
TURNS = [
    "List the files directly inside the gg_agent/ package.",
    "Which of those is the biggest, and why do you think that is?",
    "Summarize what we just established in one sentence.",
]
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not preflight():
        return 2

    from gg_agent import Agent

    with Agent(provider=PROVIDER, model=MODEL,
               event_callback=make_renderer(False)) as agent:
        banner(agent, "conversation")
        for i, turn in enumerate(TURNS, 1):
            print(f"\n› turn {i}: {turn}")
            result = agent.run(turn, keep_history=KEEP_HISTORY)
            answer(result)
            footer(result)
            print(f"[history: {len(agent.history)} messages]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
