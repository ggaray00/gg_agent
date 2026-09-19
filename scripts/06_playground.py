#!/usr/bin/env python3
"""Scratchpad — this one is yours. Edit freely, press Run, throw it away.

Everything the Agent accepts is spelled out below with its default, so you can
change one line without going back to the source to look up the name.
"""

import os

from _bootstrap import answer, banner, footer, make_renderer, preflight

# ── every knob, with its default ─────────────────────────────────────────
PROVIDER = None              # None = auto-detect from the environment
MODEL = None                 # None = the provider's default model
SYSTEM_PROMPT = None         # None = the built-in prompt
EXTRA_INSTRUCTIONS = ""      # appended to the built-in prompt — cheapest way to steer
TOOLSETS = None              # None = all. e.g. ["files"] to take the shell away
BLOCKED_TOOLS = None         # e.g. {"write_file"} for a read-only run
MAX_ITERATIONS = 50          # hard stop on loop turns
MAX_DEPTH = 2                # subagent nesting cap
MAX_TOKENS = None            # None = the provider's default
TEMPERATURE = None           # None = the provider's default
CWD = None                   # what the SYSTEM PROMPT calls the working directory.
                             # NOT a sandbox: tools resolve paths against the process
                             # cwd (the repo root, set by _bootstrap). To really move
                             # the agent, os.chdir() as well — see SCRATCH_DIR below.
VERBOSE = True               # print tool results as well as tool calls

SCRATCH_DIR = None           # set to a path to chdir there first — a real sandbox for
                             # write_file / run_shell, since both follow the process cwd

PROMPT = "What are you able to do? Answer from your actual tool list, not in general terms."
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not preflight():
        return 2

    if SCRATCH_DIR:
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        os.chdir(SCRATCH_DIR)

    from gg_agent import Agent

    with Agent(
        provider=PROVIDER,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        extra_instructions=EXTRA_INSTRUCTIONS,
        enabled_toolsets=TOOLSETS,
        blocked_tools=BLOCKED_TOOLS,
        max_iterations=MAX_ITERATIONS,
        max_depth=MAX_DEPTH,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        cwd=CWD,
        event_callback=make_renderer(VERBOSE),
    ) as agent:
        banner(agent, "playground")

        result = agent.run(PROMPT)
        answer(result)
        footer(result)

        # The whole result dict, if you want to poke at it:
        #   result["history"]  every message, including tool calls and results
        #   result["failed"], result["interrupted"], result["exit_reason"]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
