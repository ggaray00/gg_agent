#!/usr/bin/env python3
"""Ask the same question of several providers and compare.

Useful when you change the loop or a prompt and want to know whether a behaviour
is the model's or yours. Providers without credentials are skipped, not fatal.
"""

import time

from _bootstrap import load_dotenv, preflight

# ── edit me ──────────────────────────────────────────────────────────────
# (provider, model) — None model means "that provider's default"
CANDIDATES = [
    ("anthropic", None),
    ("openai", None),
    ("copilot", None),
]
PROMPT = "Reply with exactly one word: the name of the model answering this."
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    load_dotenv()
    if not preflight():
        return 2

    from gg_agent import Agent
    from gg_agent.providers import get_provider_profile

    for name, model in CANDIDATES:
        profile = get_provider_profile(name)
        if profile is None:
            print(f"\n--- {name}: unknown provider, skipping")
            continue
        if not profile.has_credentials():
            print(f"\n--- {name}: no credentials, skipping")
            continue

        print(f"\n--- {name} ---")
        started = time.time()
        try:
            with Agent(provider=name, model=model) as agent:
                result = agent.run(PROMPT)
            print(f"{result['response']}")
            print(f"[{agent.model} · {round(time.time() - started, 2)}s · "
                  f"{result['usage'].prompt_tokens}+{result['usage'].completion_tokens} tokens]")
        except Exception as exc:                       # one bad provider must not end the sweep
            print(f"failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
