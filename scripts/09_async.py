#!/usr/bin/env python3
"""The async API, and what it buys you.

The core is async all the way down: the loop, the transports, the tools and the
MCP sessions. `agent.run()` is a sync wrapper over a background event loop, kept
so scripts and tests need no ceremony. When you are already async, skip it and
await directly — and then several agents cost tasks, not threads.

DEMO = "sequential"  three questions one after another
DEMO = "concurrent"  the same three at once with asyncio.gather
DEMO = "both"        run both and compare the wall clock
"""

import asyncio
import time

from _bootstrap import make_renderer, preflight

# ── edit me ──────────────────────────────────────────────────────────────
PROVIDER = None
MODEL = None
DEMO = "both"
VERBOSE = False

QUESTIONS = [
    "In one sentence: what is an event loop?",
    "In one sentence: what is a semaphore?",
    "In one sentence: what is a coroutine?",
]
# ─────────────────────────────────────────────────────────────────────────


async def ask_one(question: str) -> tuple[str, float, str]:
    """One throwaway agent per question, closed on the way out."""
    from gg_agent import Agent

    started = time.time()
    async with Agent(provider=PROVIDER, model=MODEL,
                     event_callback=make_renderer(VERBOSE) if VERBOSE else None) as agent:
        answer = await agent.aask(question)
    return question, time.time() - started, answer.strip()


async def sequential() -> float:
    print("\n=== sequential ===")
    started = time.time()
    for question in QUESTIONS:
        _, elapsed, answer = await ask_one(question)
        print(f"  [{elapsed:5.2f}s] {answer[:90]}")
    return time.time() - started


async def concurrent() -> float:
    print("\n=== concurrent (asyncio.gather) ===")
    started = time.time()
    # Three agents, three in-flight API calls, one thread.
    results = await asyncio.gather(*(ask_one(q) for q in QUESTIONS))
    for _, elapsed, answer in results:
        print(f"  [{elapsed:5.2f}s] {answer[:90]}")
    return time.time() - started


async def main() -> int:
    if not preflight():
        return 2

    seq = await sequential() if DEMO in {"sequential", "both"} else None
    con = await concurrent() if DEMO in {"concurrent", "both"} else None

    print()
    if seq is not None:
        print(f"sequential total: {seq:5.2f}s")
    if con is not None:
        print(f"concurrent total: {con:5.2f}s")
    if seq and con:
        print(f"speedup:          {seq / con:.1f}×")
    return 0


if __name__ == "__main__":
    # asyncio.run, not the sync wrapper: this script is the async caller.
    raise SystemExit(asyncio.run(main()))
