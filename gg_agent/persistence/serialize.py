"""Message dict <-> row conversion, shared by every store.

The loop's canonical format is OpenAI-shaped dicts (see ``loop.record_assistant_message``
and ``loop.run_tool_round``). A row round-trips back to exactly that dict: keys the
loop never set are omitted rather than written back as ``None``.

Mirrors hermes-agent: hermes_state_messages.py (row mapping) +
_drop_trailing_empty_response_scaffolding
"""

from __future__ import annotations

import json
from typing import Any

SNIPPET_RADIUS = 80


def message_to_row(msg: dict[str, Any]) -> dict[str, Any]:
    content = msg.get("content")
    if content is not None and not isinstance(content, str):
        # Multimodal / structured content: stored as JSON text so it stays searchable.
        content = json.dumps(content, ensure_ascii=False)
    return {
        "role": msg["role"],
        "content": content,
        "tool_calls": msg.get("tool_calls") or None,
        "tool_call_id": msg.get("tool_call_id"),
        "tool_name": msg.get("name"),
    }


def row_to_message(row: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": row["role"]}
    if row.get("tool_call_id") is not None:
        msg["tool_call_id"] = row["tool_call_id"]
    if row.get("tool_name") is not None:
        msg["name"] = row["tool_name"]
    if row.get("content") is not None:
        msg["content"] = row["content"]
    if row.get("tool_calls"):
        tool_calls = row["tool_calls"]
        msg["tool_calls"] = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
    return msg


def repair_for_resume(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop a dangling tool round at the end of a transcript.

    A crash between "assistant asked for tools" and "every tool replied" leaves
    tool calls without answers, and every provider rejects that request. The
    assistant message that opened the round is dropped along with any partial
    replies, so the model simply gets the turn again.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        wanted = {tc.get("id") for tc in msg["tool_calls"]}
        answered = {m.get("tool_call_id") for m in messages[i + 1:] if m.get("role") == "tool"}
        return messages if wanted <= answered else messages[:i]
    return messages


def count_tool_calls(messages: list[dict[str, Any]]) -> int:
    return sum(len(m.get("tool_calls") or ()) for m in messages if m.get("role") == "assistant")


def snippet_around(content: str | None, needle: str, radius: int = SNIPPET_RADIUS) -> str:
    """±``radius`` chars around the first case-insensitive match, match in **bold**."""
    text = content or ""
    pos = text.lower().find(needle.lower()) if needle else -1
    if pos < 0:
        out = text[: radius * 2]
        return out + ("…" if len(text) > len(out) else "")
    start, end = max(0, pos - radius), min(len(text), pos + len(needle) + radius)
    return ("…" if start else "") + text[start:pos] + "**" + text[pos:pos + len(needle)] + "**" \
        + text[pos + len(needle):end] + ("…" if end < len(text) else "")
