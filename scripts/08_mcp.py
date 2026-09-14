#!/usr/bin/env python3
"""MCP servers as ordinary tools.

Reads `.mcp.json` at the repo root and connects everything enabled there. Once
connected, an MCP tool is indistinguishable from a built-in one: same registry,
same dispatch, same toolset filtering.

The default config uses `@modelcontextprotocol/server-everything`, a reference
server that needs no credentials — it just needs npx on your PATH.
"""

import asyncio

from _bootstrap import banner, footer, make_renderer, preflight

# ── edit me ──────────────────────────────────────────────────────────────
PROVIDER = None
MODEL = None
CONFIG = None            # None = .mcp.json at the repo root; or a path
VERBOSE = True
LIST_ONLY = False        # True: connect, print the tool list, don't call the model

# TOOLSETS is where MCP earns the naming scheme: each server is its own toolset,
# so this is a real grant, not a suggestion.
#   None                      every tool, built-in and MCP
#   ["mcp:everything"]        ONLY that server's tools
#   ["files", "mcp:everything"]  built-in file tools plus that server
TOOLSETS = None

PROMPT = "Use the MCP tools to add 17 and 25, then echo the phrase 'mcp works'. Report both results."
# ─────────────────────────────────────────────────────────────────────────


async def main() -> int:
    if not preflight():
        return 2

    from gg_agent import Agent
    from gg_agent.tools.mcp_tools import close_mcp_servers, load_mcp_config

    configs = load_mcp_config(CONFIG)
    if not configs:
        print("No MCP servers configured. Create .mcp.json at the repo root:\n")
        print('  {"mcpServers": {"everything": {')
        print('     "command": "npx",')
        print('     "args": ["-y", "@modelcontextprotocol/server-everything"]}}}')
        return 1
    print(f"configured servers: {', '.join(c.name + ('' if c.enabled else ' (disabled)') for c in configs)}")

    agent = Agent(provider=PROVIDER, model=MODEL, mcp=CONFIG or True,
                  enabled_toolsets=TOOLSETS, event_callback=make_renderer(VERBOSE))
    try:
        # Explicit, so the tool list is populated before the banner prints it.
        # (arun() would do this on its own at the start of the first turn.)
        status = await agent.connect_mcp()
        print()
        for name, state in status.items():
            print(f"  🔌 {name:<16} {state}")

        from gg_agent.tools.registry import registry
        mcp_tools = [n for n in registry.all_names()
                     if (registry.get(n).toolset or "").startswith("mcp:")]
        print(f"\n{len(mcp_tools)} MCP tool(s) registered:")
        for name in mcp_tools:
            print(f"    {name}")

        if LIST_ONLY:
            return 0

        banner(agent, "mcp")
        result = await agent.arun(PROMPT)
        print(f"\n{result['response']}")
        footer(result)
        return 0
    finally:
        await agent.aclose()
        # Servers are a process-wide pool shared with subagents, so closing the
        # agent does NOT stop them — that is a separate, deliberate call.
        await close_mcp_servers()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
