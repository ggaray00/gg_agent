"""Anthropic Messages transport (api_mode="anthropic_messages").

This is the transport abstraction earning its keep: the loop still hands over
OpenAI-shaped messages, and everything protocol-specific happens here —
the system prompt leaves the message list, tool results become user-role
``tool_result`` blocks, and assistant tool calls become ``tool_use`` blocks.

Mirrors hermes-agent: agent/transports/anthropic.py
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import ProviderTransport
from .streaming import StreamHooks, StreamInterrupted
from .types import NormalizedResponse, ToolCall, Usage

# A breakpoint says "cache the prompt from byte 0 through this block". The
# 5-minute TTL is the cheap one to write (1.25x vs 2x for "1h") and an agent
# loop's iterations are seconds apart, so every read refreshes it anyway.
CACHE_CONTROL = {"type": "ephemeral"}


def as_blocks(message: dict[str, Any]) -> None:
    """Render a string body as a one-element block list, in place.

    Only blocks can carry ``cache_control``, so the marked turn has to be one.
    Every turn is converted, not just that one, because the SHAPE has to be
    stable across iterations: a turn sent as a bare string on this request and
    as a block list on the next (because that is the one that got the marker)
    moves the prefix bytes, and a moved prefix is a cache miss.
    """
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        message["content"] = [{"type": "text", "text": content}]


def _append(out: list[dict[str, Any]], message: dict[str, Any]) -> None:
    """Append a converted turn, folding it into the previous one when the roles
    match. The Messages API takes alternating turns; two in a row is a 400."""
    if not out or out[-1]["role"] != message["role"]:
        out.append(message)
        return
    previous, incoming = out[-1], message["content"]
    if isinstance(previous["content"], str) and isinstance(incoming, str):
        previous["content"] = f"{previous['content']}\n\n{incoming}".strip()
        return
    as_blocks(previous)
    if isinstance(previous["content"], list):
        previous["content"].extend(
            incoming if isinstance(incoming, list) else [{"type": "text", "text": str(incoming)}])
    else:                                    # previous turn was empty: take the new one
        previous["content"] = incoming


def mark_cache_breakpoint(message: dict[str, Any]) -> bool:
    """Put a breakpoint on a message's last content block.

    Returns whether a marker was placed; an empty turn has nothing to hang one on.
    """
    as_blocks(message)
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    content[-1]["cache_control"] = dict(CACHE_CONTROL)
    return True


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
        """x-api-key for Console keys; ``Authorization: Bearer`` for providers that
        declare ``bearer_auth`` (MiniMax, CommandCode) and for Claude OAuth tokens."""
        from anthropic import AsyncAnthropic

        from ..providers.plugins.anthropic import is_oauth_token
        headers = {k: v for k, v in (getattr(profile, "default_headers", None) or {}).items() if v}
        bearer = bool(getattr(profile, "bearer_auth", False))
        oauth = not bearer and getattr(profile, "name", "") == "anthropic" and is_oauth_token(api_key)
        if oauth:
            # Subscription tokens are routed by this beta flag and a Claude Code identity.
            headers.update({"anthropic-beta": "oauth-2025-04-20", "user-agent": "claude-code/2.0.0 (external, cli)",
                            "x-app": "cli"})
        # The SDK appends /v1/messages itself.
        base_url = re.sub(r"/v1/?$", "", (base_url or "").rstrip("/")) or None
        if bearer or oauth:
            client = AsyncAnthropic(auth_token=api_key, base_url=base_url, max_retries=0,
                                    default_headers=headers or None)
            # Left unset, the SDK fills api_key from $ANTHROPIC_API_KEY and sends BOTH headers.
            client.api_key = None
            return client
        return AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=0, default_headers=headers or None)

    # ── Conversion ───────────────────────────────────────────────────────

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> tuple[str, list[dict[str, Any]]]:
        """OpenAI messages -> ``(system_prompt, anthropic_messages)``.

        Consecutive tool results are merged into one user turn, which the
        Messages API requires when the assistant emitted parallel tool calls.
        Consecutive same-role turns are merged for the same reason: the loop is
        free to produce them — context compression splices a summary in as a user
        turn, which can land next to the user turn before it — and this is the
        layer that knows the wire cannot carry them.
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
                _append(out, {"role": "assistant",
                              "content": blocks or [{"type": "text", "text": "..."}]})

            else:  # user
                _append(out, {"role": "user", "content": str(msg.get("content") or "")})

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
        """Assemble the request, and mark it up for prompt caching.

        Caching is a PREFIX match — the key is the exact bytes of the rendered
        prompt up to each breakpoint — and the render order is
        ``tools`` -> ``system`` -> ``messages``. Two breakpoints (of the four
        the API allows) cover an agent loop:

        * one on the system block, which is the same every iteration and, being
          last in the stable region, caches the tool definitions with it;
        * one on the final message, so the next iteration — which has this
          whole conversation as its prefix — reads back everything before it.

        Below the model's minimum cacheable prefix (512-4096 tokens, depending
        on the model) the markers are silently ignored, which costs nothing.
        """
        profile = params.pop("profile", None)
        cache_prompt = params.pop("cache_prompt", True)
        # Chat-Completions-side hints with no Messages API counterpart here.
        for key in ("reasoning_config", "session_id", "base_url"):
            params.pop(key, None)
        system, converted = self.convert_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted,
            # max_tokens is REQUIRED by the Messages API — unlike Chat Completions.
            "max_tokens": params.pop("max_tokens", None)
                          or (profile.get_max_tokens(model) if profile is not None else None) or 8192,
        }
        if system:
            api_kwargs["system"] = (
                [{"type": "text", "text": system, "cache_control": dict(CACHE_CONTROL)}]
                if cache_prompt else system
            )
        if tools:
            api_kwargs["tools"] = self.convert_tools(tools)
        if cache_prompt and converted:
            for message in converted:
                as_blocks(message)
            mark_cache_breakpoint(converted[-1])

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
            usage=self._usage(usage) if usage else None,
        )

    @staticmethod
    def _usage(usage: Any) -> Usage:
        """Messages API counts -> the normalized (OpenAI-convention) shape.

        ``input_tokens`` here is the UNCACHED REMAINDER, not the whole prompt:
        the real prompt size is ``input_tokens + cache_read + cache_creation``.
        Reporting it raw would silently under-count every cached turn — an
        agent that ran for an hour looking like it spent 4K input tokens.
        """
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        prompt = (getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_write
        completion = getattr(usage, "output_tokens", 0) or 0
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cached_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
