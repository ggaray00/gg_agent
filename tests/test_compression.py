"""Context compression: sizing, boundaries, pruning (A), summarizing (B), loop wiring."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.aio import run_sync
from gg_agent.compression import (
    MIN_TAIL_MESSAGES,
    align_tail_start,
    compression_threshold,
    fallback_summary,
    is_summary_message,
    note_real_usage,
    protected_tail_start,
    reclaim,
    render_for_summary,
    resolve_context_length,
    summarize_middle,
)
from gg_agent.loop import PERSISTED_KEY, mark_persisted, pending_messages
from gg_agent.prompts import SUMMARY_PREFIX
from gg_agent.transports.base import ProviderTransport
from gg_agent.transports.types import NormalizedResponse, ToolCall, Usage

SCRIPT = []
SEEN = []


class FakeTransport(ProviderTransport):
    @property
    def api_mode(self): return "compress-fake"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, messages, **kw):
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
    def convert_tools(self, tools): return tools
    def build_kwargs(self, model, messages, tools=None, **params):
        return {"model": model, "messages": self.convert_messages(messages), "tools": tools}
    def call(self, client, **api_kwargs):
        SEEN.append(api_kwargs)
        return SCRIPT.pop(0)
    def normalize_response(self, response): return response


register_transport(FakeTransport())
register_provider(ProviderProfile(name="compress-fake", api_mode="compress-fake", default_model="fake-1"))


def say(text, prompt_tokens=0):
    return NormalizedResponse(content=text, tool_calls=None, finish_reason="stop",
                              usage=Usage(prompt_tokens, 5, prompt_tokens + 5))


def call(name, args, prompt_tokens=0):
    return NormalizedResponse(content=None, finish_reason="tool_calls",
                              usage=Usage(prompt_tokens, 5, prompt_tokens + 5),
                              tool_calls=[ToolCall(id="c1", name=name, arguments=json.dumps(args))])


def transcript(result_size=4000, rounds=5, persisted=True):
    """system + first user turn + ``rounds`` tool rounds, each with a big result."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do the thing"}]
    for i in range(rounds):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read_file",
                         "content": f"line {i}\n" + "x" * result_size})
    if persisted:
        mark_persisted(messages)
    return messages


# ── Sizing ───────────────────────────────────────────────────────────────────

def test_1_context_length_resolves_by_model_then_override(monkeypatch):
    monkeypatch.delenv("GG_CONTEXT_LENGTH", raising=False)
    a = Agent(provider="compress-fake", model="claude-sonnet-5")
    assert resolve_context_length(a) == 200_000
    a.model = "totally-unknown-model"
    assert resolve_context_length(a) == 128_000            # default
    monkeypatch.setenv("GG_CONTEXT_LENGTH", "50000")
    assert resolve_context_length(a) == 50_000
    a.context_length = 4096                                # explicit wins over env
    assert resolve_context_length(a) == 4096
    print("✓ 1 context length resolution order")


def test_2_threshold_reserves_output_space_and_stays_reachable():
    a = Agent(provider="compress-fake", context_length=100_000, max_tokens=None)
    assert compression_threshold(a) == 75_000
    # max_tokens comes out of the same window before the percentage applies.
    b = Agent(provider="compress-fake", context_length=100_000, max_tokens=20_000)
    assert compression_threshold(b) == 60_000
    # A window smaller than the output reservation must still yield a usable
    # threshold rather than 0 (which would compress on every single call).
    c = Agent(provider="compress-fake", context_length=8_000, max_tokens=8_000)
    assert 0 < compression_threshold(c) < 8_000
    print("✓ 2 threshold reserves output space")


def test_3_estimator_calibrates_against_billed_tokens():
    a = Agent(provider="compress-fake")
    assert a._token_scale == 1.0
    note_real_usage(a, 1000, 1200)
    assert a._token_scale == 1.2
    note_real_usage(a, 1000, 999_999)        # clamped: bad usage must not wedge the session
    assert a._token_scale == 2.0
    note_real_usage(a, 0, 500)               # no estimate to calibrate against
    assert a._token_scale == 2.0
    print("✓ 3 estimator calibration is clamped")


# ── Boundaries ───────────────────────────────────────────────────────────────

def test_4_protected_tail_is_a_token_budget_with_a_message_floor():
    messages = transcript(result_size=40, rounds=10)
    # Generous budget: everything fits, nothing is prunable.
    assert protected_tail_start(messages, tail_tokens=100_000) == 0
    # Tight budget: only the newest messages are protected...
    cut = protected_tail_start(messages, tail_tokens=60)
    assert 0 < cut < len(messages)
    # ...but never fewer than the floor, however big those messages are.
    assert len(messages) - protected_tail_start(transcript(result_size=9000, rounds=8),
                                                tail_tokens=10) > MIN_TAIL_MESSAGES
    print("✓ 4 tail budget with message floor")


def test_5_head_and_tail_are_never_pruned():
    messages = transcript(result_size=4000, rounds=6)
    before = [m.get("content") for m in messages]
    stats = reclaim(messages, tail_tokens=500)
    assert stats["tokens"] > 0
    assert messages[0]["content"] == before[0] and messages[1]["content"] == before[1]
    for i in range(stats["tail_start"], len(messages)):
        assert messages[i].get("content") == before[i], f"tail message {i} was rewritten"
    # Something in the middle did get pruned, and it says so.
    pruned = [m for m in messages if m.get("_pruned")]
    assert pruned and "read_file result pruned" in pruned[0]["content"]
    assert pruned[0]["tool_call_id"] and pruned[0]["role"] == "tool"
    print("✓ 5 head and tail protected, middle pruned")


def test_6_tool_pairing_survives_pruning():
    messages = transcript(result_size=4000, rounds=6)
    reclaim(messages, tail_tokens=500)
    wanted = [tc["id"] for m in messages if m.get("role") == "assistant"
              for tc in m.get("tool_calls") or ()]
    answered = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
    assert wanted == answered, "every tool call must still have exactly one reply"
    print("✓ 6 tool call/reply pairing intact")


# ── The passes ───────────────────────────────────────────────────────────────

def test_7_identical_results_keep_only_the_newest_copy():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(6):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "git_status", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": "git_status",
                         "content": "clean tree\n" + "y" * 2000})
    mark_persisted(messages)
    stats = reclaim(messages, tail_tokens=400)
    results = [m for m in messages if m["role"] == "tool"]
    assert [m for m in results if "repeated git_status result" in m["content"]], \
        "duplicate results should be collapsed"
    assert "clean tree" in results[-1]["content"], "the newest copy is the one kept"
    # Every prunable copy is gone; the protected tail keeps whatever it holds.
    assert all("clean tree" not in m["content"]
               for m in messages[:stats["tail_start"]] if m["role"] == "tool")
    print("✓ 7 duplicate tool results collapsed")


def test_8_oversized_tool_call_arguments_stay_valid_json():
    messages = transcript(result_size=10, rounds=6)
    messages[2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"path": "f.py", "content": "z" * 5000})
    reclaim(messages, tail_tokens=200)
    args = messages[2]["tool_calls"][0]["function"]["arguments"]
    parsed = json.loads(args)                      # must still parse: Anthropic re-parses these
    assert parsed["path"] == "f.py"
    assert len(parsed["content"]) < 5000 and "more chars" in parsed["content"]
    print("✓ 8 tool call args shrunk, still valid JSON")


def test_9_pruning_is_idempotent():
    messages = transcript(result_size=4000, rounds=6)
    first = reclaim(messages, tail_tokens=500)["tokens"]
    second = reclaim(messages, tail_tokens=500)["tokens"]
    assert first > 0 and second == 0, "a second pass must not re-prune stubs"
    print("✓ 9 pruning is idempotent")


def test_10_pressure_pass_reaches_a_single_huge_result():
    """The message-count floor protects the very result that filled the window."""
    messages = transcript(result_size=60_000, rounds=2)
    assert reclaim(messages, tail_tokens=500)["tokens"] == 0        # all inside the floor
    reclaimed = reclaim(messages, tail_tokens=0, pressure=True)["tokens"]
    assert reclaimed > 0
    assert messages[-1]["content"].startswith("line 1"), "the round in flight is untouched"
    assert messages[3].get("_pruned"), "the older result is not"
    print("✓ 10 pressure pass reaches the huge result")


def test_11_unpersisted_messages_are_left_alone():
    """Rewriting a message the store has not accepted would persist the stub."""
    messages = transcript(result_size=4000, rounds=6, persisted=False)
    assert reclaim(messages, tail_tokens=500, durable_only=True)["tokens"] == 0
    mark_persisted(messages)
    assert reclaim(messages, tail_tokens=500, durable_only=True)["tokens"] > 0
    print("✓ 11 only durable messages are pruned")


# ── Loop wiring ──────────────────────────────────────────────────────────────

def test_12_loop_compresses_before_the_next_request():
    SEEN.clear()
    SCRIPT[:] = [call("read_file", {"path": "big.log"}), say("done")]
    a = Agent(provider="compress-fake", context_length=3_000, stream=False)
    a.history = transcript(result_size=20_000, rounds=4)[1:]     # no system row in history
    r = a.run("and now?")
    assert r["response"] == "done"
    sent = SEEN[-1]["messages"]
    assert any("pruned to save context" in str(m.get("content")) for m in sent)
    assert not any("_persisted" in m for m in sent), "internal keys never reach the provider"
    # The transcript is still valid: every call answered, in order.
    wanted = [tc["id"] for m in sent if m.get("role") == "assistant" for tc in m.get("tool_calls") or ()]
    assert wanted == [m["tool_call_id"] for m in sent if m.get("role") == "tool"]
    print("✓ 12 loop compresses before the next request")


def test_13_compression_off_leaves_the_transcript_alone():
    SEEN.clear()
    SCRIPT[:] = [say("fine")]
    a = Agent(provider="compress-fake", context_length=3_000, compress=False, stream=False)
    a.history = transcript(result_size=20_000, rounds=4)[1:]
    a.run("and now?")
    assert not any("pruned to save context" in str(m.get("content")) for m in SEEN[-1]["messages"])
    print("✓ 13 compress=False is respected")


def test_14_context_full_is_reported_once_when_nothing_is_prunable():
    events = []
    SCRIPT[:] = [say("ok")]
    a = Agent(provider="compress-fake", context_length=2_000, stream=False,
              event_callback=lambda k, p: events.append((k, p)))
    a.history = [{"role": "user", "content": "x" * 30_000, PERSISTED_KEY: True}]
    a.run("more")
    full = [p for k, p in events if k == "context_full"]
    assert len(full) == 1 and full[0]["tokens"] > full[0]["threshold"]
    print("✓ 14 unprunable context reported once")


# ── Persistence markers ──────────────────────────────────────────────────────

def test_15_pruned_messages_are_not_written_to_the_store_again():
    messages = transcript(result_size=4000, rounds=6)            # all marked durable
    assert pending_messages(messages) == []
    reclaim(messages, tail_tokens=500, durable_only=True)
    assert pending_messages(messages) == [], "rewriting a durable message must not re-queue it"
    messages.append({"role": "user", "content": "next"})
    assert pending_messages(messages) == [{"role": "user", "content": "next"}]
    print("✓ 15 compression does not re-queue durable messages")


def test_16_markers_survive_a_turn_and_only_new_messages_are_flushed():
    from gg_agent.aio import run_sync
    from gg_agent.persistence import InMemorySessionStore
    from gg_agent.persistence.users import register_user

    store = InMemorySessionStore()
    user = run_sync(register_user(store, "compress@example.com", "password123"))
    SCRIPT[:] = [say("first")]
    a = Agent(provider="compress-fake", store=store, user_id=user.id, stream=False)
    r = a.run("one")
    assert r["persisted"] == 2 and all(m.get(PERSISTED_KEY) for m in a.history)
    SCRIPT[:] = [say("second")]
    r = a.run("two")
    assert r["persisted"] == 4
    assert [m.content for m in run_sync(store.get_messages(a.session_id))] == \
        ["one", "first", "two", "second"], "nothing written twice"
    a.close()
    print("✓ 16 markers dedupe across turns")


# ── Phase B: summarizing ─────────────────────────────────────────────────────

def summary_agent(**kwargs):
    """An agent whose transcript is big enough that phase B is the only way out."""
    kwargs.setdefault("context_length", 6_000)
    return Agent(provider="compress-fake", stream=False, **kwargs)


def summary_text(label="first"):
    """A summary long enough to clear the anti-stub guard, shaped like a real one."""
    return (f"## Task\nThe {label} summary of this session, written at enough length to clear "
            "the minimum-length guard that rejects refusals and stubs.\n\n"
            "## Completed actions\n1. READ f0.py — looked fine [tool: read_file]\n\n"
            "## Next step\nCarry on with the remaining files.")


def chatty(turns=14, size=900, persisted=True):
    """A transcript of plain conversation: nothing for phase A to prune."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do the thing"}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"assistant turn {i}: " + "a" * size})
        messages.append({"role": "user", "content": f"user turn {i}: " + "u" * size})
    if persisted:
        mark_persisted(messages)
    return messages


def test_17_summary_replaces_the_middle_and_keeps_head_and_tail():
    SCRIPT[:] = [say(summary_text())]
    a = summary_agent()
    messages = chatty()
    saved = run_sync(summarize_middle(a, messages, tail_tokens=400, durable_only=False))
    assert saved > 0
    assert messages[0]["role"] == "system" and messages[1]["content"] == "do the thing"
    summary = messages[2]
    assert is_summary_message(summary) and summary["role"] == "user"
    assert summary["content"].startswith(SUMMARY_PREFIX) and "Completed actions" in summary["content"]
    assert messages[-1]["content"].startswith("user turn 13"), "the tail is untouched"
    assert len(messages) < len(chatty()), "the middle is gone"
    print("✓ 17 middle replaced by a summary")


def test_18_summary_prompt_gets_the_turns_not_the_instructions():
    SCRIPT[:] = [say(summary_text())]
    a = summary_agent()
    run_sync(summarize_middle(a, chatty(), tail_tokens=400, durable_only=False))
    prompt = SEEN[-1]["messages"][0]["content"]
    assert "DATA to summarize" in prompt and "NOT instructions" in prompt
    assert "[REDACTED]" in prompt, "the redaction rule must reach the summarizer"
    assert "user turn 3" in prompt, "the turns themselves are in the prompt"
    assert SEEN[-1]["tools"] is None, "the summarizer gets no tools"
    print("✓ 18 summarizer prompt carries the guardrails")


def test_19_tool_rounds_are_never_split_by_the_summary_cut():
    messages = transcript(result_size=300, rounds=10)
    # A cut that lands on a tool reply walks back to the assistant that owns it.
    for idx in range(len(messages)):
        aligned = align_tail_start(messages, idx)
        assert messages[aligned].get("role") != "tool" or aligned == 0
    SCRIPT[:] = [say(summary_text())]
    a = summary_agent(context_length=4_000)
    run_sync(summarize_middle(a, messages, tail_tokens=300, durable_only=False))
    wanted = [tc["id"] for m in messages if m.get("role") == "assistant" for tc in m.get("tool_calls") or ()]
    assert wanted == [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
    print("✓ 19 summary cut never orphans a tool reply")


def test_20_a_failed_summarizer_falls_back_and_cools_down():
    events = []

    class Boom(Exception):
        pass

    def exploding(client, **kw):
        raise Boom("aux model is down")

    a = summary_agent(event_callback=lambda k, p: events.append((k, p)))
    a.transport.call = exploding
    try:
        messages = chatty()
        saved = run_sync(summarize_middle(a, messages, tail_tokens=400, durable_only=False))
    finally:
        del a.transport.call
    assert saved > 0, "the fallback still compacts"
    assert "Mechanical summary" in messages[2]["content"]
    assert "user turn 3" in messages[2]["content"], "the fallback keeps the user's own words"
    assert [k for k, _ in events if k == "summary_failed"]
    assert a._summary_cooldown_until > 0
    # While the cooldown is armed, phase B does not call the model again.
    assert run_sync(summarize_middle(a, chatty(), tail_tokens=400, durable_only=False)) == 0
    print("✓ 20 summarizer failure falls back and cools down")


def test_21_an_empty_or_stub_summary_counts_as_a_failure():
    SCRIPT[:] = [say("sure!")]                       # too short to be a real summary
    a = summary_agent()
    messages = chatty()
    run_sync(summarize_middle(a, messages, tail_tokens=400, durable_only=False))
    assert "Mechanical summary" in messages[2]["content"]
    assert a._summary_cooldown_until > 0
    print("✓ 21 stub summaries are rejected")


def test_22_a_later_summary_updates_the_earlier_one():
    SCRIPT[:] = [say(summary_text("first"))]
    a = summary_agent()
    messages = chatty()
    run_sync(summarize_middle(a, messages, tail_tokens=400, durable_only=False))
    mark_persisted(messages)
    # Enough new conversation that the middle is worth a second call.
    for i in range(6):
        messages.append({"role": "assistant", "content": f"later {i} " + "b" * 900, PERSISTED_KEY: True})
        messages.append({"role": "user", "content": f"later question {i} " + "c" * 900, PERSISTED_KEY: True})
    SCRIPT[:] = [say(summary_text("second"))]
    run_sync(summarize_middle(a, messages, tail_tokens=400, durable_only=False))
    prompt = SEEN[-1]["messages"][0]["content"]
    assert "An earlier compaction produced this summary" in prompt
    assert "The first summary of this session" in prompt, \
        "the previous summary is handed over, not re-summarized"
    assert len([m for m in messages if is_summary_message(m)]) == 1, "one summary, not a pile"
    print("✓ 22 summaries update rather than accumulate")


def test_23_unpersisted_messages_are_never_summarized_away():
    SCRIPT[:] = [say(summary_text())]
    a = summary_agent()
    messages = chatty(persisted=False)
    assert run_sync(summarize_middle(a, messages, tail_tokens=400, durable_only=True)) == 0
    assert SCRIPT, "no model call when there is nothing safe to drop"
    print("✓ 23 unpersisted messages are not dropped")


def test_24_phase_b_runs_only_when_pruning_was_not_enough():
    # Tool-heavy: phase A alone gets under the threshold, so no summary call.
    SCRIPT[:] = [say("answer")]
    a = Agent(provider="compress-fake", context_length=20_000, stream=False)
    a.history = transcript(result_size=6_000, rounds=8)[1:]
    a.run("next")
    assert not any(m for m in a.history if is_summary_message(m)), "phase A was enough"
    # Conversation-heavy: nothing to prune, so phase B has to run.
    SCRIPT[:] = [say(summary_text()), say("answer")]
    b = Agent(provider="compress-fake", context_length=6_000, stream=False)
    b.history = chatty(turns=16)[1:]
    b.run("next")
    assert [m for m in b.history if is_summary_message(m)], "phase B should have run"
    print("✓ 24 phase B only runs when phase A was not enough")


def test_25_compression_that_does_not_help_is_switched_off():
    events = []
    a = summary_agent(context_length=3_000, event_callback=lambda k, p: events.append((k, p)))
    a._summary_cooldown_until = float("inf")        # phase B unavailable
    # Tool results too small to prune, conversation too big to fit: passes do nothing.
    a.history = [{"role": "user", "content": "x" * 600, PERSISTED_KEY: True},
                 {"role": "assistant", "content": "y" * 20_000, PERSISTED_KEY: True},
                 {"role": "user", "content": "z" * 600, PERSISTED_KEY: True}]
    SCRIPT[:] = [say("one"), say("two"), say("three")]
    for _ in range(3):
        a.run("again")
    assert a._compression_strikes >= 1 or [k for k, _ in events if k == "context_full"]
    print("✓ 25 useless compression stops")


def test_26_rendered_turns_are_capped_and_labelled():
    rendered = render_for_summary([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "run_shell", "arguments": json.dumps({"command": "ls"})}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "run_shell", "content": "q" * 50_000},
    ])
    assert "[user]" in rendered and "[assistant calls run_shell]" in rendered
    assert "[tool result: run_shell]" in rendered
    assert "more chars" in rendered and len(rendered) < 10_000
    print("✓ 26 turns rendered compactly for the summarizer")


def test_27_fallback_summary_keeps_the_anchors():
    messages = transcript(result_size=100, rounds=3)[2:]
    messages.insert(0, {"role": "user", "content": "please fix the parser"})
    text = fallback_summary(messages, previous="earlier: set up the repo")
    assert "Mechanical summary" in text and "earlier: set up the repo" in text
    assert "please fix the parser" in text and "read_file" in text
    assert "f0.py" in text, "file paths are the anchors worth keeping"
    print("✓ 27 fallback summary keeps user turns, tools and paths")
