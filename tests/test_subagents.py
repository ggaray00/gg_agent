"""Subagent + anthropic-transport conversion tests (no network)."""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.aio import run_sync
from gg_agent.tools.delegate_tool import _build_child
from gg_agent.tools.delegate_tool import delegate_task as _adelegate_task
from gg_agent.transports.base import ProviderTransport
from gg_agent.transports.types import NormalizedResponse, ToolCall


def delegate_task(**kw):
    """delegate_task is async now; these tests drive it from sync code."""
    return run_sync(_adelegate_task(**kw))

LOCK = threading.Lock()
SCRIPTS = {}   # keyed by first user/system marker -> list of responses

class ScriptedTransport(ProviderTransport):
    """Routes each agent to its own script based on a marker in its system prompt."""
    @property
    def api_mode(self): return "scripted"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, m, **kw): return m
    def convert_tools(self, t): return t
    def build_kwargs(self, model, messages, tools=None, **p):
        return {"model": model, "messages": messages, "tools": tools}
    def call(self, client, **kw):
        system = kw["messages"][0]["content"]
        key = "child" if "focused subagent" in system else "parent"
        with LOCK:
            return SCRIPTS[key].pop(0) if SCRIPTS[key] else \
                NormalizedResponse(content="(done)", tool_calls=None, finish_reason="stop")
    def normalize_response(self, r): return r

register_transport(ScriptedTransport())
register_provider(ProviderProfile(name="scripted", api_mode="scripted", default_model="s-1"))

# model-driven delegation, two parallel children
def test_1_model_driven_delegation_two_parallel_children():
    SCRIPTS["parent"] = [
        NormalizedResponse(content=None, finish_reason="tool_calls", tool_calls=[
            ToolCall(id="d1", name="delegate_task", arguments=json.dumps({"tasks": [
                {"goal": "count files", "context": "use run_shell"},
                {"goal": "read the readme", "context": "use read_file"},
            ]}))]),
        NormalizedResponse(content="combined answer", tool_calls=None, finish_reason="stop"),
    ]
    SCRIPTS["child"] = [
        NormalizedResponse(content="child summary A", tool_calls=None, finish_reason="stop"),
        NormalizedResponse(content="child summary B", tool_calls=None, finish_reason="stop"),
    ]
    events = []
    a = Agent(provider="scripted", event_callback=lambda k, p: events.append((k, p)))
    r = a.run("do two things")
    assert r["response"] == "combined answer", r
    tool_msg = json.loads([m for m in r["history"] if m["role"] == "tool"][0]["content"])
    assert tool_msg["subagents"] == 2 and tool_msg["completed"] == 2, tool_msg
    summaries = sorted(e["summary"] for e in tool_msg["results"])
    assert summaries == ["child summary A", "child summary B"], summaries
    assert [e["task_index"] for e in tool_msg["results"]] == [0, 1]
    # The parent's transcript contains ONLY the delegation call + summary, never the child's turns.
    assert len(r["history"]) == 4, r["history"]
    assert any(k == "delegate_start" for k, _ in events)
    assert any(p.get("depth") == 1 for _, p in events), "child events should report depth=1"
    print("✓ 1 parallel delegation, parent context isolated")


# depth cap — a leaf child cannot delegate
def test_2_depth_cap_a_leaf_child_cannot_delegate():
    parent = Agent(provider="scripted", max_depth=1)
    child = _build_child(parent, {"goal": "g"}, 0)
    assert child.depth == 1 and not child.can_delegate()
    assert "delegate_task" not in [t["name"] for t in child.tool_definitions()]
    assert "delegate_task" in [t["name"] for t in parent.tool_definitions()]
    assert "focused subagent" in child.system_prompt and "g" in child.system_prompt
    # ...and calling it anyway is refused
    assert "depth cap" in delegate_task(goal="x", parent_agent=child)["error"]
    print("✓ 2 depth cap blocks leaf delegation")


# nested delegation allowed under max_depth=2
def test_3_nested_delegation_allowed_under_max_depth_2():
    p2 = Agent(provider="scripted", max_depth=2)
    c2 = _build_child(p2, {"goal": "g"}, 0)
    assert c2.can_delegate() and "delegate_task" in [t["name"] for t in c2.tool_definitions()]
    assert "Orchestrator Role" in c2.system_prompt
    g2 = _build_child(c2, {"goal": "gg"}, 0)
    assert g2.depth == 2 and not g2.can_delegate()
    print("✓ 3 nested orchestrator, grandchild is a leaf")


# validation + fan-out cap
def test_4_validation_fan_out_cap():
    p = Agent(provider="scripted")
    assert "goal" in delegate_task(parent_agent=p)["error"]
    assert "non-empty" in delegate_task(tasks=[{"context": "x"}], parent_agent=p)["error"]
    assert "Too many tasks" in delegate_task(tasks=[{"goal": str(i)} for i in range(9)], parent_agent=p)["error"]
    assert "requires a parent agent" in delegate_task(goal="x")["error"]
    print("✓ 4 delegation validation")


# a crashing child returns an entry, doesn't kill the parent
def test_5_a_crashing_child_returns_an_entry_doesn_t_kill_the_parent():
    SCRIPTS["child"] = []
    class Boom(ScriptedTransport):
        @property
        def api_mode(self): return "boom"
        def call(self, client, **kw):
            if "focused subagent" in kw["messages"][0]["content"]:
                raise ValueError("child exploded")
            return NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")
    register_transport(Boom())
    register_provider(ProviderProfile(name="boom", api_mode="boom", default_model="b"))
    import gg_agent.loop as L
    L.MAX_API_RETRIES, orig = 1, L.MAX_API_RETRIES
    pb = Agent(provider="boom")
    out = delegate_task(tasks=[{"goal": "will fail"}], parent_agent=pb)
    assert out["results"][0]["status"] == "failed", out
    assert "child exploded" in out["results"][0]["error"], out
    assert pb._children == [], "children must be detached after the batch"
    L.MAX_API_RETRIES = orig
    print("✓ 5 child failure isolated")


# Anthropic transport message conversion
def test_6_anthropic_transport_message_conversion():
    from gg_agent.transports.anthropic import AnthropicTransport
    t = AnthropicTransport()
    system, msgs = t.convert_messages([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "let me look",
         "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "run_shell", "arguments": '{"command":"ls"}'}},
                        {"id": "t2", "type": "function", "function": {"name": "list_dir", "arguments": '{}'}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "run_shell", "content": "a.py"},
        {"role": "tool", "tool_call_id": "t2", "name": "list_dir", "content": "b/"},
        {"role": "assistant", "content": "found it"},
    ])
    assert system == "SYS"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"], msgs
    assert [b["type"] for b in msgs[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert msgs[1]["content"][1]["input"] == {"command": "ls"}
    # both tool results merged into ONE user turn, as the Messages API requires
    assert len(msgs[2]["content"]) == 2 and msgs[2]["content"][0]["tool_use_id"] == "t1"
    kw = t.build_kwargs("claude-sonnet-5", [{"role": "user", "content": "x"}],
                        tools=[{"name": "f", "description": "d", "parameters": {"type": "object"}}])
    assert kw["max_tokens"] == 8192 and kw["tools"][0]["input_schema"] == {"type": "object"}
    assert t.map_finish_reason("tool_use") == "tool_calls" and t.map_finish_reason("end_turn") == "stop"
    print("✓ 6 anthropic conversion")


# chat_completions build_kwargs + profile hooks
def test_7_chat_completions_build_kwargs_profile_hooks():
    from gg_agent.providers import get_provider_profile
    from gg_agent.transports.chat_completions import ChatCompletionsTransport
    c = ChatCompletionsTransport()
    kw = c.build_kwargs("gpt-4.1", [{"role": "user", "content": "x", "_internal": "drop me"}],
                        tools=[{"name": "f", "description": "d", "parameters": {}}],
                        profile=get_provider_profile("openrouter"))
    assert "_internal" not in kw["messages"][0]
    assert kw["tools"][0] == {"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}
    assert kw["extra_body"]["provider"]["require_parameters"] is True
    assert kw["tool_choice"] == "auto"
    ol = c.build_kwargs("q", [{"role": "user", "content": "x"}], profile=get_provider_profile("ollama"))
    assert ol["temperature"] == 0.0 and "tools" not in ol
    print("✓ 7 chat_completions kwargs + profile hooks")

