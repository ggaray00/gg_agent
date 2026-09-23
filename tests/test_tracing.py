"""Langfuse tracing: the span tree a turn produces, checked against the real SDK
with an in-memory OTel exporter (no network, no keys)."""
import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.tracing import NoopTracer, Tracer, _usage_details, resolve_tracer
from gg_agent.transports.base import ProviderTransport
from gg_agent.transports.types import NormalizedResponse, ToolCall, Usage

SCRIPT = []


class ScriptedTransport(ProviderTransport):
    @property
    def api_mode(self): return "fake_traced"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, messages, **kw): return messages
    def convert_tools(self, tools): return tools
    def build_kwargs(self, model, messages, tools=None, **params):
        return {"model": model, "messages": list(messages), "tools": tools}
    def call(self, client, **api_kwargs):
        item = SCRIPT.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    def normalize_response(self, response): return response


register_transport(ScriptedTransport())
register_provider(ProviderProfile(name="fake-traced", api_mode="fake_traced", default_model="fake-1"))


@pytest.fixture
def traced():
    """A Tracer over a real Langfuse client whose spans land in memory."""
    langfuse = pytest.importorskip("langfuse")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    # Langfuse keeps one client per public key, so each test needs its own key.
    client = langfuse.Langfuse(public_key=f"pk-test-{uuid.uuid4().hex}", secret_key="sk-test", base_url="http://localhost:1",
                               tracer_provider=TracerProvider(), span_exporter=exporter)
    tracer = Tracer(client)

    def spans():
        client.flush()
        return {s.name: s for s in exporter.get_finished_spans()}, exporter.get_finished_spans()
    yield tracer, spans
    client.shutdown()


def _attr(span, key):
    return span.attributes.get(f"langfuse.observation.{key}")


def test_turn_generation_and_tool_spans_nest(traced):
    tracer, spans = traced
    SCRIPT[:] = [
        NormalizedResponse(content=None, finish_reason="tool_calls", usage=Usage(20, 8, 28, cached_tokens=5),
                           tool_calls=[ToolCall(id="c1", name="run_shell",
                                                arguments=json.dumps({"command": "echo hi"}))]),
        NormalizedResponse(content="it printed hi", tool_calls=None, finish_reason="stop",
                           usage=Usage(30, 4, 34)),
    ]
    a = Agent(provider="fake-traced", tracing=tracer, store=False, stream=False)
    r = a.run("run echo hi")
    a.close()
    assert r["response"] == "it printed hi"

    by_name, all_spans = spans()
    root, tool = by_name["gg-agent"], by_name["run_shell"]
    gens = [s for s in all_spans if s.name == "fake-traced/fake-1"]
    assert len(gens) == 2 and len(all_spans) == 4, [s.name for s in all_spans]
    assert {s.context.trace_id for s in all_spans} == {root.context.trace_id}
    assert all(s.parent.span_id == root.context.span_id for s in (tool, *gens))

    assert _attr(root, "type") == "agent" and _attr(tool, "type") == "tool"
    assert _attr(gens[0], "type") == "generation" and _attr(gens[0], "model.name") == "fake-1"
    assert "it printed hi" in _attr(root, "output")
    assert "hi" in _attr(tool, "output")
    assert json.loads(_attr(gens[0], "usage_details")) == {"input": 15, "output": 8, "input_cached_tokens": 5}
    assert root.attributes.get("session.id") == a.session_id
    # Loop-private markers never reach the trace.
    assert "_persisted" not in _attr(gens[1], "input")


def test_failed_call_marks_generation_and_turn_as_error(traced):
    tracer, spans = traced
    err = ValueError("bad request")
    err.status_code = 400
    SCRIPT[:] = [err]
    a = Agent(provider="fake-traced", tracing=tracer, store=False, stream=False)
    r = a.run("hi")
    assert r["failed"]
    by_name, _ = spans()
    assert _attr(by_name["fake-traced/fake-1"], "level") == "ERROR"
    assert _attr(by_name["gg-agent"], "level") == "ERROR"


def test_tracing_off_by_default_and_inherited_by_children():
    a = Agent(provider="fake-traced", store=False)
    assert isinstance(a.tracer, NoopTracer) and not a.tracer.enabled
    parent = type("P", (), {"tracer": object()})()
    assert resolve_tracer(None, parent) is parent.tracer
    assert not resolve_tracer(False, parent).enabled


def test_anthropic_usage_keys_split_cache_reads_and_writes():
    u = Usage(prompt_tokens=100, completion_tokens=10, cached_tokens=60, cache_write_tokens=30)
    assert _usage_details(u, "anthropic_messages") == {
        "input": 10, "output": 10, "cache_read_input_tokens": 60, "cache_creation_input_tokens": 30}


def test_subagent_turn_nests_under_delegate_tool_span(traced):
    tracer, spans = traced
    SCRIPT[:] = [
        NormalizedResponse(content=None, finish_reason="tool_calls", tool_calls=[
            ToolCall(id="d1", name="delegate_task", arguments=json.dumps({"goal": "say done"}))]),
        NormalizedResponse(content="child done", tool_calls=None, finish_reason="stop"),
        NormalizedResponse(content="parent done", tool_calls=None, finish_reason="stop"),
    ]
    a = Agent(provider="fake-traced", tracing=tracer, store=False, stream=False)
    assert a.run("delegate it")["response"] == "parent done"
    by_name, all_spans = spans()
    child, delegate = by_name["subagent"], by_name["delegate_task"]
    assert child.parent.span_id == delegate.context.span_id
    assert {s.context.trace_id for s in all_spans} == {by_name["gg-agent"].context.trace_id}
    assert "child done" in _attr(child, "output")
