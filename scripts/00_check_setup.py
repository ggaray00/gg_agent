#!/usr/bin/env python3
"""Sanity check — the only script here that makes NO API call.

Run this first. It answers: which providers can I use, what tools are loaded,
and where is each credential coming from?
"""

from _bootstrap import ROOT, load_dotenv  # noqa: F401  (also fixes sys.path)


def main() -> int:
    loaded = load_dotenv()
    print(f"repo root : {ROOT}")
    print(f".env      : {'loaded ' + ', '.join(loaded) if loaded else 'none found (using shell env)'}")

    from gg_agent.providers import list_providers
    from gg_agent.tools.registry import discover_builtin_tools, registry

    print("\n--- providers ---")
    for profile in list_providers():
        mark = "✓" if profile.has_credentials() else " "
        print(f" {mark} {profile.name:<12} {profile.api_mode:<20} {profile.default_model}")
    print("\n ✓ = auto-detected. The others still work if you name them explicitly.")

    print("\n--- credential detail ---")
    for profile in list_providers():
        try:
            status = profile.credential_status()
        except Exception as exc:                        # a probe may hit the network
            status = f"error: {exc}"
        print(f" {profile.name:<12} {status}")

    print("\n--- tools ---")
    discover_builtin_tools()
    for name in registry.all_names():
        entry = registry.get(name)
        avail = "" if entry.available() else "  (unavailable here)"
        print(f" {entry.emoji} {name:<16} [{entry.toolset}]{avail}")

    print("\n--- toolsets ---")
    for name, description in registry.toolsets().items():
        print(f" {name:<12} {description}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
