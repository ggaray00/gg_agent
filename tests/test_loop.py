"""End-to-end loop test against a scripted fake provider (no network)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.transports.base import ProviderTransport
from gg_agent.transports.types import NormalizedResponse, ToolCall, Usage

SCRIPT = []   # list of NormalizedResponse to return in order
SEEN = []     # kwargs the transport received

class FakeTransport(ProviderTransport):
    @property
    def api_mode(self): return "fake"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, messages, **kw): return messages
    def convert_tools(self, tools): return tools
    def build_kwargs(self, model, messages, tools=None, **params):
        return {"model": model, "messages": list(messages), "tools": tools}
    def call(self, client, **api_kwargs):
        SEEN.append(api_kwargs)
        return SCRIPT.pop(0)
    def normalize_response(self, response): return response

register_transport(FakeTransport())
register_provider(ProviderProfile(name="fake", api_mode="fake", default_model="fake-1"))

# plain text answer, no tools
def test_1_plain_text_answer_no_tools():
    SCRIPT[:] = [NormalizedResponse(content="hello there", tool_calls=None, finish_reason="stop",
                                    usage=Usage(10, 5, 15))]
    a = Agent(provider="fake")
    r = a.run("hi")
    assert r["response"] == "hello there", r
    assert r["api_calls"] == 1 and r["tool_calls"] == 0 and r["exit_reason"] == "final_response"
    assert r["usage"].total_tokens == 15
    assert [m["role"] for m in r["history"]] == ["user", "assistant"], r["history"]
    assert SEEN[0]["messages"][0]["role"] == "system"
    print("✓ 1 plain answer")


# tool call round-trip
def test_2_tool_call_round_trip():
    SEEN.clear()
    SCRIPT[:] = [
        NormalizedResponse(content=None, finish_reason="tool_calls", usage=Usage(20, 8, 28),
            tool_calls=[ToolCall(id="c1", name="run_shell", arguments=json.dumps({"command": "echo hi"}))]),
        NormalizedResponse(content="it printed hi", tool_calls=None, finish_reason="stop", usage=Usage(30, 4, 34)),
    ]
    a = Agent(provider="fake")
    r = a.run("run echo hi")
    assert r["response"] == "it printed hi", r
    assert r["api_calls"] == 2 and r["tool_calls"] == 1
    assert [m["role"] for m in r["history"]] == ["user", "assistant", "tool", "assistant"], r["history"]
    assert "hi" in r["history"][2]["content"], r["history"][2]
    assert r["history"][2]["tool_call_id"] == "c1"
    assert r["usage"].total_tokens == 62
    print("✓ 2 tool round-trip")


# parallel tool calls keep request order
def test_3_parallel_tool_calls_keep_request_order():
    SEEN.clear()
    SCRIPT[:] = [
        NormalizedResponse(content=None, finish_reason="tool_calls", tool_calls=[
            ToolCall(id="a", name="run_shell", arguments=json.dumps({"command": "echo AAA"})),
            ToolCall(id="b", name="run_shell", arguments=json.dumps({"command": "echo BBB"})),
            ToolCall(id="c", name="list_dir", arguments=json.dumps({"path": str(Path(__file__).resolve().parent.parent)})),
        ]),
        NormalizedResponse(content="done", tool_calls=None, finish_reason="stop"),
    ]
    a = Agent(provider="fake")
    r = a.run("three things")
    tools = [m for m in r["history"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["a", "b", "c"], tools
    assert "AAA" in tools[0]["content"] and "BBB" in tools[1]["content"] and "gg_agent" in tools[2]["content"]
    print("✓ 3 parallel tools, ordered replies")


# unknown tool / bad JSON args come back as errors, not crashes
def test_4_unknown_tool_bad_json_args_come_back_as_errors_not_crashes():
    SCRIPT[:] = [
        NormalizedResponse(content=None, finish_reason="tool_calls", tool_calls=[
            ToolCall(id="x", name="nope", arguments="{}"),
            ToolCall(id="y", name="run_shell", arguments="{not json"),
        ]),
        NormalizedResponse(content="recovered", tool_calls=None, finish_reason="stop"),
    ]
    a = Agent(provider="fake")
    r = a.run("break it")
    tools = [m for m in r["history"] if m["role"] == "tool"]
    assert "Unknown tool" in tools[0]["content"], tools[0]
    assert "invalid JSON" in tools[1]["content"], tools[1]
    assert r["response"] == "recovered"
    print("✓ 4 tool errors returned to model")


# max_iterations cap
def test_5_max_iterations_cap():
    SCRIPT[:] = [NormalizedResponse(content=None, finish_reason="tool_calls", tool_calls=[
        ToolCall(id=f"i{i}", name="run_shell", arguments=json.dumps({"command": "true"}))]) for i in range(5)]
    a = Agent(provider="fake", max_iterations=3)
    r = a.run("loop forever")
    assert r["api_calls"] == 3 and r["exit_reason"] == "max_iterations", r
    print("✓ 5 iteration cap")


# API failure is retried then surfaced
def test_6_api_failure_is_retried_then_surfaced():
    class FlakyTransport(FakeTransport):
        @property
        def api_mode(self): return "flaky"
        def call(self, client, **kw):
            raise RuntimeError("rate limit exceeded")
    register_transport(FlakyTransport())
    register_provider(ProviderProfile(name="flaky", api_mode="flaky", default_model="f"))
    import gg_agent.loop as L
    L.MAX_API_RETRIES = 2
    orig_sleep = L.time.sleep; L.time.sleep = lambda s: None
    a = Agent(provider="flaky")
    r = a.run("hi")
    assert r["failed"] and r["exit_reason"] == "api_error" and "rate limit" in r["response"], r
    L.time.sleep = orig_sleep; L.MAX_API_RETRIES = 4
    print("✓ 6 retry + failure surfaced")


# history persists across turns
def test_7_history_persists_across_turns():
    SCRIPT[:] = [NormalizedResponse(content="one", tool_calls=None, finish_reason="stop"),
                 NormalizedResponse(content="two", tool_calls=None, finish_reason="stop")]
    a = Agent(provider="fake")
    a.run("first"); r = a.run("second")
    assert [m["role"] for m in a.history] == ["user","assistant","user","assistant"], a.history
    assert SEEN[-1]["messages"][1]["content"] == "first"
    a.reset(); assert a.history == []
    print("✓ 7 multi-turn history")

