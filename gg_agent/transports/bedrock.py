"""AWS Bedrock Converse transport (api_mode="bedrock_converse").

One request shape for every Bedrock model family (Claude, Llama, Nova,
Mistral...). The loop's OpenAI-shaped messages become Converse messages:
the system prompt moves to ``system``, assistant tool calls become ``toolUse``
blocks, and tool results become ``toolResult`` blocks on a user turn.

boto3 is synchronous, so the call runs in a worker thread. Not streamed: the
loop falls back to the blocking path for transports without ``call_stream``.

Mirrors hermes-agent: agent/transports/bedrock.py, agent/bedrock_adapter.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlparse

from .base import ProviderTransport
from .types import NormalizedResponse, ToolCall, Usage

_CACHE_POINT = {"cachePoint": {"type": "default"}}


def _region_from_base_url(base_url: str) -> str | None:
    host = urlparse(base_url or "").hostname or ""
    parts = host.split(".")
    return parts[1] if len(parts) > 2 and parts[0] == "bedrock-runtime" else None


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(p if isinstance(p, str) else str(p.get("text") or "")
                         for p in content if isinstance(p, (str, dict)))
    return "" if content is None else str(content)


def _append(out: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    """Converse wants alternating turns; fold a same-role turn into the previous one."""
    if not blocks:
        return
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": blocks})


class BedrockConverseTransport(ProviderTransport):
    _STOP_REASON_MAP = {
        "end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
        "guardrail_intervened": "content_filter", "content_filtered": "content_filter",
    }

    @property
    def api_mode(self) -> str:
        return "bedrock_converse"

    def build_client(self, *, api_key: str, base_url: str, profile: Any) -> Any:
        import boto3  # AWS_BEARER_TOKEN_BEDROCK, when set, is read by botocore itself
        return boto3.client("bedrock-runtime", region_name=_region_from_base_url(base_url))

    async def aclose_client(self, client: Any) -> None:
        close = getattr(client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    # ── Conversion ───────────────────────────────────────────────────────

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> tuple[list[dict], list[dict]]:
        system: list[dict[str, Any]] = []
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                text = _text_of(msg.get("content")).strip()
                if text:
                    system.append({"text": text})
            elif role == "tool":
                _append(out, "user", [{"toolResult": {
                    "toolUseId": msg.get("tool_call_id"),
                    "content": [{"text": _text_of(msg.get("content")) or "(empty)"}]}}])
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                text = _text_of(msg.get("content")).strip()
                if text:
                    blocks.append({"text": text})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    blocks.append({"toolUse": {"toolUseId": tc.get("id"), "name": fn.get("name"),
                                               "input": args if isinstance(args, dict) else {"value": args}}})
                _append(out, "assistant", blocks)
            else:
                text = _text_of(msg.get("content")).strip()
                if text:
                    _append(out, "user", [{"text": text}])
        return system, out

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"toolSpec": {"name": t["name"], "description": t.get("description") or t["name"],
                              "inputSchema": {"json": t.get("parameters") or {"type": "object", "properties": {}}}}}
                for t in tools or []]

    def build_kwargs(self, model: str, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None, **params) -> dict[str, Any]:
        profile = params.pop("profile", None)
        cache_prompt = params.pop("cache_prompt", True)
        for key in ("reasoning_config", "session_id", "base_url", "tool_choice"):
            params.pop(key, None)
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        if profile is not None:
            clean = profile.prepare_messages(clean)
        system, converted = self.convert_messages(clean)

        # Claude and Nova on Bedrock honour cache points; other families reject them.
        caches = cache_prompt and any(fam in model.lower() for fam in ("anthropic", "claude", "nova"))
        api_kwargs: dict[str, Any] = {"modelId": model, "messages": converted}
        if system:
            api_kwargs["system"] = system + ([dict(_CACHE_POINT)] if caches else [])
        if tools:
            api_kwargs["toolConfig"] = {"tools": self.convert_tools(tools)}
        inference: dict[str, Any] = {}
        max_tokens = params.pop("max_tokens", None) or (profile.get_max_tokens(model) if profile else None)
        if max_tokens:
            inference["maxTokens"] = max_tokens
        temperature = params.pop("temperature", None)
        if temperature is not None:
            inference["temperature"] = temperature
        if inference:
            api_kwargs["inferenceConfig"] = inference
        api_kwargs.update({k: v for k, v in params.items() if v is not None})
        return api_kwargs

    async def call(self, client: Any, **api_kwargs) -> Any:
        return await asyncio.to_thread(client.converse, **api_kwargs)

    # ── Normalization ────────────────────────────────────────────────────

    def normalize_response(self, response: Any) -> NormalizedResponse:
        blocks = ((response.get("output") or {}).get("message") or {}).get("content") or []
        texts: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in blocks:
            if "text" in block:
                texts.append(block["text"])
            elif "toolUse" in block:
                use = block["toolUse"]
                tool_calls.append(ToolCall(id=use.get("toolUseId"), name=use.get("name", ""),
                                           arguments=json.dumps(use.get("input") or {})))
            elif "reasoningContent" in block:
                text = ((block["reasoningContent"] or {}).get("reasoningText") or {}).get("text")
                if text:
                    reasoning.append(text)
        u = response.get("usage") or {}
        cache_read, cache_write = u.get("cacheReadInputTokens", 0) or 0, u.get("cacheWriteInputTokens", 0) or 0
        # Like Anthropic, inputTokens is the uncached remainder: fold the cache back in.
        prompt = (u.get("inputTokens", 0) or 0) + cache_read + cache_write
        completion = u.get("outputTokens", 0) or 0
        return NormalizedResponse(
            content="".join(texts) or None,
            tool_calls=tool_calls or None,
            finish_reason=self.map_finish_reason(response.get("stopReason")),
            reasoning="\n".join(reasoning) or None,
            usage=Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
                        cached_tokens=cache_read, cache_write_tokens=cache_write),
        )
