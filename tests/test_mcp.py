"""MCP integration: config, naming, schema translation, failure handling.

Uses a fake session rather than a real server, so it is fast and offline. The
one live-server check is in scripts/08_mcp.py.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent.tools import mcp_tools as M
from gg_agent.tools.registry import registry


class FakeTool:
    """Mirrors the SDK model, snake_case as of mcp 2.x."""
    def __init__(self, name, description="", schema=None):
        self.name, self.description = name, description
        self.input_schema = schema or {"type": "object", "properties": {"x": {"type": "string"}}}

class LegacyTool:
    """The older camelCase spelling — both must work."""
    def __init__(self, name):
        self.name, self.description = name, "legacy"
        self.inputSchema = {"type": "object", "properties": {"y": {"type": "integer"}}}

class FakeBlock:
    def __init__(self, text): self.type, self.text = "text", text

class FakeResult:
    def __init__(self, text="ok", is_error=False, structured=None):
        self.content = [FakeBlock(text)]
        self.is_error, self.structured_content = is_error, structured


# config parsing, including ${VAR} expansion and disabled servers
def test_1_config_parsing(tmp_path=None, monkeypatch=None):
    import os
    import tempfile
    os.environ["GG_TEST_TOKEN"] = "s3cret"
    blob = {"mcpServers": {
        "alpha": {"command": "echo", "args": ["hi"], "env": {"TOKEN": "${GG_TEST_TOKEN}"}},
        "beta": {"url": "https://example.test/mcp", "headers": {"A": "b"}},
        "off": {"command": "echo", "enabled": False},
    }}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / ".mcp.json"
        path.write_text(json.dumps(blob))
        configs = {c.name: c for c in M.load_mcp_config(path)}
    assert configs["alpha"].env["TOKEN"] == "s3cret", "env ${VAR} must expand"
    assert configs["alpha"].kind == "stdio" and configs["beta"].kind == "http"
    assert configs["off"].enabled is False
    assert M.load_mcp_config(Path("/nonexistent/.mcp.json")) == []
    print("✓ 1 config parsing, env expansion, http vs stdio")


# a server that cannot start is reported, not raised
def test_2_unstartable_server_is_reported():
    cfg = M.MCPServerConfig(name="ghost", command="definitely-not-a-real-binary-xyz")
    assert "not found on PATH" in cfg.problem()
    assert M.MCPServerConfig(name="empty").problem() is not None

    async def go():
        pool = M.MCPPool()
        return await pool.connect([cfg])
    status = asyncio.run(go())
    assert "unavailable" in status["ghost"] and "not found" in status["ghost"]
    print("✓ 2 broken server reported, never raised")


# tools register namespaced, with the schema carried across
def test_3_registration_namespacing_and_schema():
    pool = M.MCPPool()
    server = M.MCPServer(M.MCPServerConfig(name="demo", command="echo"))
    server.tools = [FakeTool("search", "Find things"), LegacyTool("old-style")]
    server.session = object()                      # pretend initialized
    pool.servers["demo"] = server
    M._register_server_tools(pool, server)

    entry = registry.get("demo__search")
    assert entry is not None, "tool must be registered under <server>__<tool>"
    assert entry.toolset == "mcp:demo", entry.toolset
    assert entry.schema["parameters"]["properties"] == {"x": {"type": "string"}}
    # camelCase inputSchema must survive too — silently losing it would make the
    # model call the tool with no arguments.
    legacy = registry.get("demo__old-style")
    assert legacy.schema["parameters"]["properties"] == {"y": {"type": "integer"}}, legacy.schema
    print("✓ 3 namespaced registration, both schema spellings")


# a dead server's tools drop out of the grant instead of erroring
def test_4_dead_server_tools_are_hidden():
    entry = registry.get("demo__search")
    # pool.is_alive("demo") is False: the fake server has no live task.
    assert not entry.available(), "a dead server's tools must report unavailable"
    names = [t["name"] for t in registry.get_definitions()]
    assert "demo__search" not in names, "unavailable tools must leave the tool list"
    print("✓ 4 dead server drops out of the tool list")


# result rendering: text, structured, errors
def test_5_result_rendering():
    assert M._render_result(FakeResult("hello")) == "hello"
    err = M._render_result(FakeResult("boom", is_error=True))
    assert json.loads(err)["error"] == "boom"
    structured = M._render_result(FakeResult("ignored", structured={"n": 42}))
    assert json.loads(structured) == {"n": 42}

    class Camel:                                   # older SDK spelling
        content, isError, structuredContent = [FakeBlock("legacy")], True, None
    assert json.loads(M._render_result(Camel()))["error"] == "legacy"
    print("✓ 5 text / structured / error results")


# names are sanitized to what the function-calling APIs accept
def test_6_name_sanitization():
    assert M._sanitize("weird name!/v2") == "weird_name__v2"
    assert len(M._sanitize("x" * 200)) <= 48
    print("✓ 6 tool names sanitized")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
