"""Streaming: transports, tool-call reassembly, think-tag scrubbing, loop rules (no network)."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.stream_delivery import StreamDelivery, ThinkScrubber
from gg_agent.transports.anthropic import AnthropicTransport
from gg_agent.transports.base import ProviderTransport
from gg_agent.transports.chat_completions import ChatCompletionsTransport
from gg_agent.transports.streaming import (
    StreamDropped,
    StreamHooks,
    StreamInterrupted,
    ToolCallAccumulator,
)
from gg_agent.transports.types import NormalizedResponse, ToolCall, Usage


class Recorder:
    """StreamHooks that remember what they were given."""

    def __init__(self, interrupt_after: int | None = None):
        self.text, self.reasoning, self.tools = [], [], []
        self.interrupt_after = interrupt_after

    def hooks(self) -> StreamHooks:
        return StreamHooks(
            on_text=self.text.append, on_reasoning=self.reasoning.append,
            on_tool_start=self.tools.append,
            is_interrupted=lambda: self.interrupt_after is not None and len(self.text) >= self.interrupt_after)


# ── Chat Completions ─────────────────────────────────────────────────────────

def chunk(content=None, tool_calls=None, finish=None, reasoning=None, usage=None, choices=True):
    delta = NS(content=content, tool_calls=tool_calls, reasoning_content=reasoning)
    return NS(choices=[NS(delta=delta, finish_reason=finish)] if choices else [], usage=usage)


def tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class FakeStream:
    def __init__(self, chunks):
        self.chunks, self.closed = list(chunks), False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self.chunks:
            yield c

    async def close(self):
        self.closed = True


def openai_client(result):
    async def create(**kwargs):
        create.kwargs = kwargs
        return result
    return NS(chat=NS(completions=NS(create=create))), create


def test_chat_stream_text_and_usage():
    stream = FakeStream([
        chunk(reasoning="hmm "), chunk(content="Hel"), chunk(content="lo"), chunk(finish="stop"),
        chunk(choices=False, usage=NS(prompt_tokens=3, completion_tokens=2, total_tokens=5)),
    ])
    client, create = openai_client(stream)
    rec = Recorder()
    r = asyncio.run(ChatCompletionsTransport().call_stream(client, rec.hooks(), model="m", messages=[]))
    assert create.kwargs["stream"] is True and create.kwargs["stream_options"] == {"include_usage": True}
    assert rec.text == ["Hel", "lo"] and rec.reasoning == ["hmm "]
    assert r.content == "Hello" and r.reasoning == "hmm " and r.finish_reason == "stop"
    assert r.usage.total_tokens == 5 and r.tool_calls is None
    assert stream.closed


def test_chat_stream_tool_call_fragments():
    stream = FakeStream([
        chunk(content="Let me check."),
        chunk(tool_calls=[tc(0, id="a", name="run_shell", args='{"comm')]),
        chunk(tool_calls=[tc(0, args='and": "ls"}')]),
        chunk(tool_calls=[tc(1, id="b", name="read_file", args='{"path": "x"}')]),
        chunk(content=" trailing"),                       # after a tool call: kept, not shown
        chunk(finish="tool_calls"),
    ])
    client, _ = openai_client(stream)
    rec = Recorder()
    r = asyncio.run(ChatCompletionsTransport().call_stream(client, rec.hooks(), model="m", messages=[]))
    assert rec.text == ["Let me check."] and rec.tools == ["run_shell", "read_file"]
    assert [(t.id, t.name, json.loads(t.arguments)) for t in r.tool_calls] == [
        ("a", "run_shell", {"command": "ls"}), ("b", "read_file", {"path": "x"})]
    assert r.finish_reason == "tool_calls" and r.content == "Let me check. trailing"


def test_accumulator_ollama_reuses_index_zero():
    acc = ToolCallAccumulator()
    assert acc.feed(tc(0, id="a", name="f", args="{}")) == "f"
    assert acc.feed(tc(0, id="b", name="g", args='{"x": 1}')) == "g"
    assert acc.feed(tc(0, name="g")) is None            # resent full name: no new announce
    calls, truncated = acc.materialize()
    assert [(c.id, c.name, c.arguments) for c in calls] == [("a", "f", "{}"), ("b", "g", '{"x": 1}')]
    assert not truncated


def test_chat_stream_drop_mid_tool_call_raises():
    stream = FakeStream([chunk(tool_calls=[tc(0, id="a", name="f", args='{"half')])])
    client, _ = openai_client(stream)
    try:
        asyncio.run(ChatCompletionsTransport().call_stream(client, Recorder().hooks(), model="m", messages=[]))
    except StreamDropped:
        pass
    else:
        raise AssertionError("expected StreamDropped")


def test_chat_stream_interrupt():
    stream = FakeStream([chunk(content="a"), chunk(content="b"), chunk(content="c")])
    client, _ = openai_client(stream)
    rec = Recorder(interrupt_after=1)
    try:
        asyncio.run(ChatCompletionsTransport().call_stream(client, rec.hooks(), model="m", messages=[]))
    except StreamInterrupted:
        pass
    else:
        raise AssertionError("expected StreamInterrupted")
    assert rec.text == ["a"] and stream.closed


def test_chat_stream_non_iterator_response_is_replayed():
    whole = NS(choices=[NS(message=NS(content="all at once", tool_calls=None), finish_reason="stop")], usage=None)
    client, _ = openai_client(whole)
    rec = Recorder()
    r = asyncio.run(ChatCompletionsTransport().call_stream(client, rec.hooks(), model="m", messages=[]))
    assert r.content == "all at once" and rec.text == ["all at once"]


# ── Anthropic ────────────────────────────────────────────────────────────────

class FakeAnthropicStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self.events:
            yield e

    async def get_final_message(self):
        return self.final


def test_anthropic_stream_routes_events():
    events = [
        NS(type="content_block_start", content_block=NS(type="thinking")),
        NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking="plan")),
        NS(type="content_block_start", content_block=NS(type="text")),
        NS(type="content_block_delta", delta=NS(type="text_delta", text="Sure.")),
        NS(type="text", text="Sure."),                     # SDK convenience event: ignored
        NS(type="content_block_start", content_block=NS(type="tool_use", name="run_shell")),
        NS(type="content_block_delta", delta=NS(type="input_json_delta", partial_json='{"c')),
    ]
    final = NS(content=[NS(type="thinking", thinking="plan"), NS(type="text", text="Sure."),
                        NS(type="tool_use", id="t1", name="run_shell", input={"command": "ls"})],
               stop_reason="tool_use", usage=NS(input_tokens=4, output_tokens=6, cache_read_input_tokens=0))
    client = NS(messages=NS(stream=lambda **kw: FakeAnthropicStream(events, final)))
    rec = Recorder()
    r = asyncio.run(AnthropicTransport().call_stream(client, rec.hooks(), model="m", messages=[]))
    assert rec.text == ["Sure."] and rec.reasoning == ["plan"] and rec.tools == ["run_shell"]
    assert r.finish_reason == "tool_calls" and r.tool_calls[0].arguments == '{"command": "ls"}'
    assert r.usage.total_tokens == 10


# ── Think scrubbing ──────────────────────────────────────────────────────────

def scrub(pieces):
    s, vis, rea = ThinkScrubber(), [], []
    for p in pieces:
        v, r = s.feed(p)
        vis.append(v)
        rea.append(r)
    v, r = s.flush()
    return "".join(vis) + v, "".join(rea) + r


def test_think_tag_split_across_chunks():
    assert scrub(["<thi", "nk>weigh", "ing</th", "ink>\n\nAnswer"]) == ("\n\nAnswer", "weighing")


def test_think_tag_mentioned_mid_line_is_text():
    text = "Models emit <think> tags sometimes."
    assert scrub([text[:15], text[15:]]) == (text, "")


def test_think_tag_at_line_start_later():
    assert scrub(["Intro\n<thinking>x</thinking>rest"]) == ("Intro\nrest", "x")


def test_delivery_strips_leading_newlines_and_breaks_segments():
    events = []
    d = StreamDelivery(lambda kind, **p: events.append((kind, p.get("text"))))
    d.fire_text("<think>r</think>\n\nOne")
    d.finish()
    d.segment_break()
    d.begin_attempt()
    d.fire_text("Two")
    assert events == [("reasoning_delta", "r"), ("stream_delta", "One"), ("stream_break", None),
                      ("stream_delta", "\n\nTwo")]


# ── The loop ─────────────────────────────────────────────────────────────────

SCRIPT = []      # per call: (text_pieces, NormalizedResponse | Exception)
CALLS = []       # "stream" | "call"


class FakeStreamingTransport(ProviderTransport):
    supports_streaming = True

    @property
    def api_mode(self): return "fake_stream"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, messages, **kw): return messages
    def convert_tools(self, tools): return tools
    def build_kwargs(self, model, messages, tools=None, **params):
        return {"model": model, "messages": list(messages)}

    def call(self, client, **api_kwargs):
        CALLS.append("call")
        _, outcome = SCRIPT.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def call_stream(self, client, hooks, **api_kwargs):
        CALLS.append("stream")
        pieces, outcome = SCRIPT.pop(0)
        for p in pieces:
            if hooks.is_interrupted():
                raise StreamInterrupted()
            hooks.on_text(p)
        for t in (outcome.tool_calls or []) if isinstance(outcome, NormalizedResponse) else []:
            hooks.on_tool_start(t.name)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def normalize_response(self, response): return response


register_transport(FakeStreamingTransport())
register_provider(ProviderProfile(name="fake_stream", api_mode="fake_stream", default_model="fs-1"))


def final(text):
    return NormalizedResponse(content=text, tool_calls=None, finish_reason="stop", usage=Usage(1, 1, 2))


def run_agent(script, **kw):
    SCRIPT[:] = script
    CALLS.clear()
    events = []
    agent = Agent(provider="fake_stream", event_callback=lambda k, p: events.append((k, p)), **kw)
    return agent, agent.run("go"), events


def deltas(events):
    return [p["text"] for k, p in events if k == "stream_delta"]


def test_loop_streams_text_and_marks_result():
    _, r, events = run_agent([(["Hi ", "there"], final("Hi there"))], stream=True)
    assert deltas(events) == ["Hi ", "there"] and r["streamed"] and r["response"] == "Hi there"
    assert CALLS == ["stream"]


def test_loop_break_before_tools():
    tool = NormalizedResponse(content="Checking.", finish_reason="tool_calls",
                              tool_calls=[ToolCall(id="c1", name="run_shell",
                                                   arguments=json.dumps({"command": "echo hi"}))])
    _, r, events = run_agent([(["Checking."], tool), (["Done."], final("Done."))], stream=True)
    kinds = [k for k, _ in events if k in {"stream_delta", "tool_gen_start", "stream_break", "tool_start"}]
    assert kinds == ["stream_delta", "tool_gen_start", "stream_break", "tool_start", "stream_delta"], kinds
    assert deltas(events) == ["Checking.", "\n\nDone."] and r["response"] == "Done."


def test_loop_no_stream_flag_uses_call():
    _, r, events = run_agent([([], final("plain"))], stream=False)
    assert CALLS == ["call"] and not r["streamed"] and deltas(events) == []


def test_loop_env_var_disables_streaming(monkeypatch):
    monkeypatch.setenv("GG_STREAM", "0")
    _, r, _ = run_agent([([], final("plain"))])
    assert CALLS == ["call"] and not r["streamed"]


def test_loop_falls_back_when_streaming_unsupported():
    agent, r, _ = run_agent([([], RuntimeError("Streaming is not supported for this model")),
                             ([], final("fallback"))], stream=True)
    assert CALLS == ["stream", "call"] and r["response"] == "fallback" and not r["streamed"]
    assert agent._stream_disabled


def test_loop_keeps_partial_text_instead_of_retrying():
    _, r, events = run_agent([(["half an ans"], ConnectionError("connection reset")),
                              (["never"], final("never"))], stream=True)
    assert CALLS == ["stream"], CALLS
    assert r["response"] == "half an ans" and r["streamed"] and r["exit_reason"] == "final_response"
    assert any(k == "stream_error" for k, _ in events)


def test_loop_retries_when_nothing_was_shown(monkeypatch):
    monkeypatch.setattr("gg_agent.loop.random.random", lambda: 0.0)
    monkeypatch.setattr("gg_agent.loop.asyncio.sleep", _instant)
    _, r, _ = run_agent([([], ConnectionError("connection reset")), (["ok"], final("ok"))], stream=True)
    assert CALLS == ["stream", "stream"] and r["response"] == "ok"


async def _instant(_):
    return None


def test_loop_interrupt_keeps_partial_answer():
    agent_box = {}
    SCRIPT[:] = [(["first ", "second"], final("first second"))]
    CALLS.clear()

    def on_event(kind, payload):
        if kind == "stream_delta":
            agent_box["agent"].interrupt()

    agent = Agent(provider="fake_stream", stream=True, event_callback=on_event)
    agent_box["agent"] = agent
    r = agent.run("go")
    assert r["interrupted"] and r["exit_reason"] == "interrupted"
    assert r["response"] == "first " and r["streamed"]
    assert r["history"][-1] == {"role": "assistant", "content": "first "}


def test_subagents_do_not_stream():
    parent = Agent(provider="fake_stream", stream=True)
    child = Agent(provider="fake_stream", stream=True, depth=1, parent=parent)
    assert parent.streaming and not child.streaming
