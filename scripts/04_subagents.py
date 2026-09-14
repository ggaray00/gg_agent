#!/usr/bin/env python3
"""Subagents, two ways.

MODE = "model"    the model sees delegate_task in its tool list and decides to fan out
MODE = "explicit" you call the fan-out primitive from Python; no model in the orchestrator seat
MODE = "both"     run one after the other

Children are sibling tasks on one event loop, so this script is async — the
explicit form awaits `delegate_task` directly.
"""

import asyncio

from _bootstrap import banner, footer, make_renderer, preflight

# ── edit me ──────────────────────────────────────────────────────────────
PROVIDER = None
MODEL = None
MODE = "both"            # "model" | "explicit" | "both"
MAX_DEPTH = 2            # how deep subagents may nest before delegation is revoked

MODEL_DRIVEN_PROMPT = (
    "Use subagents to answer both in parallel, then combine: "
    "(a) how many .py files are under the current directory, and "
    "(b) what the largest file in the current directory is."
)

EXPLICIT_TASKS = [
    {"goal": "Count the lines of Python in ./gg_agent and report the total.",
     "context": "Use run_shell. Answer with just the number and how you got it."},
    {"goal": "List every tool module under ./gg_agent/tools and describe each in one line.",
     "context": "Read the files; do not guess."},
]
# ─────────────────────────────────────────────────────────────────────────


async def run_model_driven(agent) -> None:
    print("\n=== model-driven delegation ===")
    result = await agent.arun(MODEL_DRIVEN_PROMPT)
    print(f"\n{result['response']}")
    footer(result)


async def run_explicit(agent) -> None:
    from gg_agent.tools.delegate_tool import delegate_task

    print("\n=== explicit fan-out ===")
    result = await delegate_task(tasks=EXPLICIT_TASKS, parent_agent=agent)
    if "error" in result:
        print(f"error: {result['error']}")
        return
    for entry in result["results"]:
        print(f"\n--- task {entry['task_index']} [{entry['status']}] ---\n{entry['summary']}")


async def main() -> int:
    if not preflight():
        return 2

    from gg_agent import Agent

    async with Agent(provider=PROVIDER, model=MODEL, max_depth=MAX_DEPTH,
                     event_callback=make_renderer(False)) as agent:
        banner(agent, f"subagents · mode={MODE}")
        if MODE in {"model", "both"}:
            await run_model_driven(agent)
        if MODE in {"explicit", "both"}:
            await run_explicit(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
