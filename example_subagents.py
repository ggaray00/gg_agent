#!/usr/bin/env python3
"""Two ways to use subagents.

    python example_subagents.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gg_agent import Agent  # noqa: E402
from gg_agent.tools.delegate_tool import delegate_task  # noqa: E402


def render(kind, payload):
    if kind in {"api_call", "tool_start", "delegate_start"}:
        print("  " * payload["depth"] + f"[{kind}] "
              + str(payload.get("name") or payload.get("model") or payload.get("count")))


async def main() -> int:
    agent = Agent(event_callback=render, max_depth=2)

    # 1. Let the MODEL decide to delegate — it sees delegate_task in its tool list.
    print("\n=== model-driven delegation ===")
    print(await agent.aask(
        "Use subagents to answer both in parallel, then combine: "
        "(a) how many .py files are under the current directory, and "
        "(b) what the largest file in the current directory is."
    ))

    # 2. Call delegation DIRECTLY from Python — the fan-out primitive, no model
    #    in the orchestrator seat.
    print("\n=== explicit fan-out ===")
    result = await delegate_task(
        tasks=[
            {"goal": "Count the lines of Python in ./gg_agent and report the total.",
             "context": "Use run_shell. Answer with just the number and how you got it."},
            {"goal": "List every tool module under ./gg_agent/tools and describe each in one line.",
             "context": "Read the files; do not guess."},
        ],
        parent_agent=agent,
    )
    for entry in result["results"]:
        print(f"\n--- task {entry['task_index']} [{entry['status']}] ---\n{entry['summary']}")

    await agent.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
