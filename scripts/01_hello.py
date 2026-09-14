#!/usr/bin/env python3
"""Smallest possible round trip: one question, one answer.

Edit PROMPT and press Run.
"""

from _bootstrap import banner, footer, preflight

# ── edit me ──────────────────────────────────────────────────────────────
PROVIDER = None          # None = auto-detect. Or "anthropic", "openai", "ollama", ...
MODEL = None             # None = the provider's default
PROMPT = "In two sentences, what is an agentic loop?"
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not preflight():
        return 2

    from gg_agent import Agent

    with Agent(provider=PROVIDER, model=MODEL) as agent:
        banner(agent, "hello")
        result = agent.run(PROMPT)
        print(result["response"])
        footer(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
