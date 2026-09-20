"""Prompt caching: breakpoint placement, prefix stability, and usage accounting.

The expensive failure mode here is silent — requests keep succeeding, the bill
is just higher — so these tests assert the two things nothing else would catch:
that the markers land where they are supposed to, and that a growing
conversation keeps re-sending a byte-identical prefix.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent.transports.anthropic import AnthropicTransport
from gg_agent.transports.chat_completions import ChatCompletionsTransport
from gg_agent.transports.types import Usage

TOOLS = [{"name": "run_shell", "description": "run a command", "parameters": {"type": "object"}}]

# One agent turn that has already made a tool call, as the loop would hand it over.
HISTORY = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "what is here?"},
    {"role": "assistant", "content": "looking",
     "tool_calls": [{"id": "t1", "type": "function",
                     "function": {"name": "run_shell", "arguments": '{"command":"ls"}'}}]},
    {"role": "tool", "tool_call_id": "t1", "name": "run_shell", "content": "a.py"},
]


def breakpoints(api_kwargs):
    """Every (location, block index) carrying a cache_control marker."""
    found = []
    for i, block in enumerate(api_kwargs.get("system") or []):
        if isinstance(block, dict) and "cache_control" in block:
            found.append(("system", i))
    for i, message in enumerate(api_kwargs["messages"]):
        for j, block in enumerate(message.get("content") or []):
            if isinstance(block, dict) and "cache_control" in block:
                found.append((f"messages[{i}]", j))
    return found


def strip_markers(api_kwargs):
    """The rendered prompt with breakpoints removed.

    A marker moves every request by design; it is not an invalidator, and
    previously-marked blocks stay cache hits. What must not move is anything
    else, so the markers come out before diffing.
    """
    clean = copy.deepcopy(api_kwargs)
    for block in clean.get("system") or []:
        if isinstance(block, dict):
            block.pop("cache_control", None)
    for message in clean["messages"]:
        for block in message.get("content") or []:
            if isinstance(block, dict):
                block.pop("cache_control", None)
    return clean


# breakpoints land on the system block and the conversation tail
def test_1_breakpoint_placement():
    kw = AnthropicTransport().build_kwargs("claude-opus-5", HISTORY, tools=TOOLS)

    # System became a block list so it can carry a marker; tools render before
    # it, so that one breakpoint covers the tool definitions too.
    assert kw["system"] == [{"type": "text", "text": "SYS",
                             "cache_control": {"type": "ephemeral"}}], kw["system"]
    assert kw["tools"][0]["name"] == "run_shell"

    marks = breakpoints(kw)
    assert marks == [("system", 0), ("messages[2]", 0)], marks
    # messages[2] is the merged tool_result turn — the last message.
    assert len(kw["messages"]) == 3
    assert kw["messages"][2]["content"][0]["type"] == "tool_result"
    # The API allows 4 per request; two leaves room and keeps the loop simple.
    assert len(marks) <= 4
    print("✓ 1 breakpoints on system + conversation tail")


# a plain user turn is marked too, and becomes a block list to carry it
def test_2_string_turn_becomes_blocks():
    kw = AnthropicTransport().build_kwargs(
        "claude-opus-5", [{"role": "system", "content": "SYS"},
                          {"role": "user", "content": "hi"}])
    assert kw["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}], kw["messages"]
    print("✓ 2 string turn converted and marked")


# the prefix a growing conversation re-sends must be byte-identical
def test_3_prefix_is_stable_across_iterations():
    t = AnthropicTransport()
    first = t.build_kwargs("claude-opus-5", HISTORY, tools=TOOLS)
    # Next loop iteration: the model answered, so two more turns are appended.
    grown = HISTORY + [
        {"role": "assistant", "content": "one python file"},
        {"role": "user", "content": "which one?"},
    ]
    second = t.build_kwargs("claude-opus-5", grown, tools=TOOLS)

    a, b = strip_markers(first), strip_markers(second)
    assert a["system"] == b["system"], "system moved between iterations"
    assert a["tools"] == b["tools"], "tool definitions moved between iterations"
    overlap = len(a["messages"])
    assert b["messages"][:overlap] == a["messages"], (
        "the earlier turns did not re-render byte-identically:\n"
        f"{json.dumps(a['messages'], indent=2)}\n---\n"
        f"{json.dumps(b['messages'][:overlap], indent=2)}")
    # And the new tail is where the marker moved to.
    assert breakpoints(second) == [("system", 0), ("messages[4]", 0)], breakpoints(second)
    print("✓ 3 prefix stable as the conversation grows")


# opting out leaves the request exactly as it was before caching existed
def test_4_opt_out():
    kw = AnthropicTransport().build_kwargs(
        "claude-opus-5", HISTORY, tools=TOOLS, cache_prompt=False)
    assert kw["system"] == "SYS", kw["system"]
    assert breakpoints(kw) == []
    assert kw["messages"][0]["content"] == "what is here?", "shape changed with caching off"
    print("✓ 4 cache_prompt=False is a plain request")


# input_tokens is the UNCACHED remainder — the cache fields have to be added back
def test_5_anthropic_usage_counts_the_whole_prompt():
    class RawUsage:
        input_tokens = 300
        output_tokens = 50
        cache_read_input_tokens = 9000
        cache_creation_input_tokens = 700

    class Response:
        content = []
        stop_reason = "end_turn"
        usage = RawUsage()

    u = AnthropicTransport().normalize_response(Response()).usage
    assert u.prompt_tokens == 10000, u          # 300 + 9000 + 700, not 300
    assert u.cached_tokens == 9000 and u.cache_write_tokens == 700, u
    assert u.total_tokens == 10050, u
    # Summing across a turn keeps the split.
    total = u + u
    assert (total.prompt_tokens, total.cached_tokens, total.cache_write_tokens) == (20000, 18000, 1400)
    print("✓ 5 anthropic usage counts cached tokens")


# the OpenAI-compatible providers cache with no request parameter; read their reports
def test_6_chat_completions_usage_and_no_leak():
    class Details:
        cached_tokens = 800

    class OpenAIUsage:
        prompt_tokens = 1000
        completion_tokens = 20
        total_tokens = 1020
        prompt_tokens_details = Details()

    u = Usage.from_openai(OpenAIUsage())
    # cached_tokens is a discount on prompt_tokens, not an addition to it.
    assert (u.prompt_tokens, u.cached_tokens, u.total_tokens) == (1000, 800, 1020), u

    # DeepSeek reports the same thing under a different top-level name.
    deepseek = {"prompt_tokens": 500, "completion_tokens": 10, "total_tokens": 510,
                "prompt_cache_hit_tokens": 384}
    assert Usage.from_openai(deepseek).cached_tokens == 384
    assert Usage.from_openai(None) == Usage()

    # cache_prompt is Anthropic-only; forwarding it to an OpenAI endpoint is a 400.
    kw = ChatCompletionsTransport().build_kwargs(
        "gpt-4.1", [{"role": "user", "content": "x"}], cache_prompt=True)
    assert "cache_prompt" not in kw, kw
    print("✓ 6 chat_completions usage + cache_prompt not forwarded")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
