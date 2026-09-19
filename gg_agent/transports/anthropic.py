"""Anthropic Messages transport (api_mode="anthropic_messages").

This is the transport abstraction earning its keep: the loop still hands over
OpenAI-shaped messages, and everything protocol-specific happens here —
the system prompt leaves the message list, tool results become user-role
``tool_result`` blocks, and assistant tool calls become ``tool_use`` blocks.

Mirrors hermes-agent: agent/transports/anthropic.py
"""

from __future__ import annotations

import json
from typing import Any

from .base import ProviderTransport
from .streaming import StreamHooks, StreamInterrupted
from .types import NormalizedResponse, ToolCall, Usage


class AnthropicTransport(ProviderTransport):
    supports_streaming = True

    # Anthropic stop_reason vocabulary -> OpenAI finish_reason.
    _STOP_REASON_MAP = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "refusal": "content_filter",
    }

    @property
    def api_mode(self) -> str:
        return "anthropic_messages"

    def build_client(self, *, api_key: str, base_url: str, profile: Any) -> Any:
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=api_key, base_url=base_url or None, max_retries=0)

    # ── Conversion ───────────────────────────────────────────────────────

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> tuple[str, list[dict[str, Any]]]:
        """OpenAI messages -> ``(system_prompt, anthropic_messages)``.

        Consecutive tool results are merged into one user turn, which the
        Messages API requires when the assistant emitted parallel tool calls.
        """
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                system_parts.append(str(msg.get("content") or ""))

            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": str(msg.get("content") or ""),
                }
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})

            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": str(msg["content"])})
                for tc in msg.get("tool_calls") or []:
                    fn = tc["function"] if isinstance(tc, dict) else tc.function
                    name = fn["name"] if isinstance(fn, dict) else fn.name
                    raw_args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                    try:
                        parsed = json.loads(raw_args or "{}")
                    except (TypeError, ValueError):
                        parsed = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"] if isinstance(tc, dict) else tc.id,
                        "name": name,
                        "input": parsed if isinstance(parsed, dict) else {},
                    })
                # An empty assistant turn is rejected by the API.
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "..."}]})

            else:  # user
                out.append({"role": "user", "content": str(msg.get("content") or "")})

        return "\n\n".join(p for p in system_parts if p), out

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
            }
            for t in tools or []
        ]

    def build_kwargs(self, model: str, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None, **params) -> dict[str, Any]:
        profile = params.pop("profile", None)
        system, converted = self.convert_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted,
            # max_tokens is REQUIRED by the Messages API — unlike Chat Completions.
            "max_tokens": params.pop("max_tokens", None)
                          or getattr(profile, "default_max_tokens", None) or 8192,
        }
        if system:
            api_kwargs["system"] = system
        if tools:
            api_kwargs["tools"] = self.convert_tools(tools)

        temperature = params.pop("temperature", None)
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        api_kwargs.update({k: v for k, v in params.items() if v is not None})
        return api_kwargs

    async def call(self, client: Any, **api_kwargs) -> Any:
        return await client.messages.create(**api_kwargs)

    async def call_stream(self, client: Any, hooks: StreamHooks, **api_kwargs) -> NormalizedResponse:
        """Route the SSE events to the hooks; let the SDK assemble the message.

        Tool-use JSON is not accumulated here — ``get_final_message()`` already
        does that — so this only decides what the user gets to see early."""
        in_tool = False
        text_blocks = 0
        async with client.messages.stream(**api_kwargs) as stream:
            async for event in stream:
                if hooks.is_interrupted():
                    raise StreamInterrupted()
                kind = getattr(event, "type", None)
                if kind == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        in_tool = True
                        hooks.on_tool_start(block.name)
                    elif block.type == "text":
                        # normalize_response joins text blocks with "\n"; match it.
                        if text_blocks and not in_tool:
                            hooks.on_text("\n")
                        text_blocks += 1
                elif kind == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta" and not in_tool:
                        hooks.on_text(delta.text)
                    elif delta.type == "thinking_delta":
                        hooks.on_reasoning(delta.thinking)
            final = await stream.get_final_message()
        return self.normalize_response(final)

    # ── Normalization ────────────────────────────────────────────────────

    def normalize_response(self, response: Any) -> NormalizedResponse:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif kind == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id, name=block.name,
                    arguments=json.dumps(block.input or {}),
                ))

        usage = getattr(response, "usage", None)
        return NormalizedResponse(
            content="\n".join(text_parts) or None,
            tool_calls=tool_calls or None,
            finish_reason=self.map_finish_reason(getattr(response, "stop_reason", None)),
            reasoning="\n".join(thinking_parts) or None,
            usage=Usage(
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=(getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
                cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ) if usage else None,
        )
