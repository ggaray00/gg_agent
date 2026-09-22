"""OpenAI Responses API transport (api_mode="codex_responses").

Serves the providers that are Responses-first or Responses-only: xAI, Meta
Model API, Ramp Router, Actual, and the ChatGPT-backed Codex endpoint. The loop
still hands over OpenAI Chat-shaped messages; this transport turns them into
Responses ``input`` items and folds the ``output`` items back into a
``NormalizedResponse``:

    system            -> ``instructions``
    user / assistant  -> message items
    assistant tool_calls -> ``function_call`` items
    tool              -> ``function_call_output`` items

Requests are stateless (``store: false``): the whole transcript goes out every
time, exactly as with Chat Completions, so nothing depends on server-side state.

Mirrors hermes-agent: agent/transports/codex.py, agent/codex_responses_adapter.py
"""

from __future__ import annotations

import json
from typing import Any

from ..providers.base import OMIT_TEMPERATURE
from .base import ProviderTransport
from .streaming import StreamHooks, StreamInterrupted, aclose_quietly
from .types import NormalizedResponse, ToolCall, Usage

# Transport controls a profile may return from ``build_responses_extras``.
_STREAM_ONLY = "_stream_only"
_OMIT_MAX_OUTPUT_TOKENS = "_omit_max_output_tokens"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute of an SDK model, or key of a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _content_parts(content: Any, *, role: str) -> Any:
    """Chat content -> Responses content. Strings pass through; block lists map
    ``text`` / ``image_url`` onto input_text / input_image (output_text for assistant)."""
    if not isinstance(content, list):
        return "" if content is None else str(content)
    text_type = "output_text" if role == "assistant" else "input_text"
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": text_type, "text": part})
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append({"type": text_type, "text": str(part.get("text") or "")})
        elif isinstance(part, dict) and part.get("type") == "image_url" and role != "assistant":
            url = part.get("image_url")
            parts.append({"type": "input_image", "image_url": url.get("url") if isinstance(url, dict) else url})
    return parts


class ResponsesTransport(ProviderTransport):
    supports_streaming = True

    # Response.status -> OpenAI finish_reason.
    _STOP_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "stop", "cancelled": "stop"}

    @property
    def api_mode(self) -> str:
        return "codex_responses"

    def build_client(self, *, api_key: str, base_url: str, profile: Any) -> Any:
        from openai import AsyncOpenAI
        headers = {k: v for k, v in (getattr(profile, "default_headers", None) or {}).items() if v}
        client_headers = getattr(profile, "client_headers", None)
        if callable(client_headers):          # headers derived from the live credential (Codex)
            headers.update(client_headers(api_key))
        return AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url or None,
                           default_headers=headers or None, max_retries=0)

    # ── Conversion ───────────────────────────────────────────────────────

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> tuple[str, list[dict[str, Any]]]:
        """OpenAI messages -> ``(instructions, input_items)``."""
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                instructions.append(str(msg.get("content") or ""))
            elif role == "tool":
                items.append({"type": "function_call_output", "call_id": msg.get("tool_call_id"),
                              "output": str(msg.get("content") or "")})
            elif role == "assistant":
                content = _content_parts(msg.get("content"), role="assistant")
                if content:
                    items.append({"role": "assistant", "content": content})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    items.append({"type": "function_call", "call_id": tc.get("id"),
                                  "name": fn.get("name"), "arguments": fn.get("arguments") or "{}"})
            elif role == "user":
                items.append({"role": "user", "content": _content_parts(msg.get("content"), role="user")})
        return "\n\n".join(p for p in instructions if p).strip(), items

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Registry schemas ({name, description, parameters}) -> Responses function tools (flat)."""
        return [{"type": "function", "name": t["name"], "description": t.get("description", ""),
                 "parameters": t.get("parameters") or {"type": "object", "properties": {}}, "strict": False}
                for t in tools or []]

    @staticmethod
    def _reasoning(profile: Any, model: str, reasoning_config: dict | None) -> dict[str, Any] | None:
        """``reasoning`` field from the user's config, clamped to what the profile
        declares for this model. Unset config = the provider's default (no field)."""
        from ..reasoning_effort import clamp_effort
        if not isinstance(reasoning_config, dict):
            return None
        supported = profile.supported_reasoning_efforts(model) if profile is not None else None
        if supported == ():
            return None                       # this model takes no reasoning fields at all
        disabled = reasoning_config.get("enabled") is False or reasoning_config.get("effort") == "none"
        if disabled:
            return {"effort": "none"} if supported is None or "none" in supported else None
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if not effort:
            return None
        return {"effort": clamp_effort(effort, supported)}

    def build_kwargs(self, model: str, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None, **params) -> dict[str, Any]:
        profile = params.pop("profile", None)
        params.pop("cache_prompt", None)      # Responses endpoints cache prefixes on their own
        reasoning_config = params.pop("reasoning_config", None)
        session_id = params.pop("session_id", None)
        base_url = params.pop("base_url", None)
        messages = self.convert_messages_prepared(messages, profile)
        instructions, items = self.convert_messages(messages)

        api_kwargs: dict[str, Any] = {"model": model, "input": items, "store": False}
        # Some backends (Codex) reject a request with no instructions at all.
        api_kwargs["instructions"] = instructions or "You are a helpful assistant."
        if tools:
            api_kwargs["tools"] = self.convert_tools(tools)
            api_kwargs["tool_choice"] = params.pop("tool_choice", "auto")
            api_kwargs["parallel_tool_calls"] = True

        reasoning = self._reasoning(profile, model, reasoning_config)
        if reasoning:
            api_kwargs["reasoning"] = reasoning

        extras = profile.build_responses_extras(model=model, session_id=session_id, base_url=base_url,
                                                reasoning_config=reasoning_config) if profile is not None else {}
        extras = dict(extras or {})

        temperature = params.pop("temperature", None)
        if profile is not None and profile.fixed_temperature is not None:
            temperature = profile.fixed_temperature
        if temperature is not None and temperature is not OMIT_TEMPERATURE:
            api_kwargs["temperature"] = temperature

        max_tokens = params.pop("max_tokens", None) or (profile.get_max_tokens(model) if profile else None)
        if max_tokens and not extras.pop(_OMIT_MAX_OUTPUT_TOKENS, False):
            api_kwargs["max_output_tokens"] = max_tokens
        extras.pop(_OMIT_MAX_OUTPUT_TOKENS, None)

        extra_headers = {**(api_kwargs.get("extra_headers") or {}), **(extras.pop("extra_headers", None) or {})}
        api_kwargs.update(extras)
        if extra_headers:
            api_kwargs["extra_headers"] = extra_headers
        api_kwargs.update({k: v for k, v in params.items() if v is not None})
        return api_kwargs

    @staticmethod
    def convert_messages_prepared(messages: list[dict[str, Any]], profile: Any) -> list[dict[str, Any]]:
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        return profile.prepare_messages(clean) if profile is not None else clean

    # ── Calls ────────────────────────────────────────────────────────────

    async def call(self, client: Any, **api_kwargs) -> Any:
        if api_kwargs.pop(_STREAM_ONLY, False):
            # Stream-only backend: stream anyway and hand back the assembled response.
            return await self.call_stream(client, StreamHooks(), **api_kwargs)
        return await client.responses.create(**api_kwargs)

    async def call_stream(self, client: Any, hooks: StreamHooks, **api_kwargs) -> NormalizedResponse:
        api_kwargs.pop(_STREAM_ONLY, None)
        stream = await client.responses.create(**api_kwargs, stream=True)
        if not hasattr(stream, "__aiter__"):           # endpoint ignored stream=True
            response = self.normalize_response(stream)
            if response.content:
                hooks.on_text(response.content)
            return response

        final: Any = None
        done_items: list[Any] = []
        saw_tool = False
        try:
            async for event in stream:
                if hooks.is_interrupted():
                    raise StreamInterrupted()
                kind = _get(event, "type", "")
                if kind == "response.output_text.delta":
                    if not saw_tool:
                        hooks.on_text(_get(event, "delta", "") or "")
                elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                    hooks.on_reasoning(_get(event, "delta", "") or "")
                elif kind == "response.output_item.added":
                    item = _get(event, "item")
                    if _get(item, "type") == "function_call":
                        saw_tool = True
                        hooks.on_tool_start(_get(item, "name", "") or "")
                elif kind == "response.output_item.done":
                    done_items.append(_get(event, "item"))
                elif kind in ("response.completed", "response.incomplete", "response.failed"):
                    final = _get(event, "response")
                elif kind == "error":
                    raise RuntimeError(f"Responses stream error: {_get(event, 'message') or event}")
        finally:
            await aclose_quietly(stream)
        if final is None and not done_items:
            raise ConnectionError("Responses stream connection dropped before completion")
        return self.normalize_response(final, fallback_items=done_items)

    # ── Normalization ────────────────────────────────────────────────────

    def normalize_response(self, response: Any, fallback_items: list[Any] | None = None) -> NormalizedResponse:
        output = list(_get(response, "output", None) or []) if response is not None else []
        # Some backends (Codex) stream the items but leave the final ``output`` empty.
        if not output and fallback_items:
            output = [i for i in fallback_items if i is not None]
        texts: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in output:
            kind = _get(item, "type")
            if kind == "message":
                for part in _get(item, "content", None) or []:
                    if _get(part, "type") in ("output_text", "text"):
                        texts.append(_get(part, "text", "") or "")
                    elif _get(part, "type") == "refusal":
                        texts.append(_get(part, "refusal", "") or "")
            elif kind == "function_call":
                args = _get(item, "arguments", "{}")
                tool_calls.append(ToolCall(
                    id=_get(item, "call_id") or _get(item, "id"), name=_get(item, "name", ""),
                    arguments=args if isinstance(args, str) else json.dumps(args)))
            elif kind == "reasoning":
                for part in (_get(item, "summary", None) or []) + (_get(item, "content", None) or []):
                    text = _get(part, "text")
                    if text:
                        reasoning.append(text)

        status = _get(response, "status", "completed") if response is not None else "completed"
        finish = "tool_calls" if tool_calls else self.map_finish_reason(status)
        if status == "incomplete":
            details = _get(response, "incomplete_details")
            finish = "content_filter" if _get(details, "reason") == "content_filter" else "length"
        return NormalizedResponse(
            content="".join(texts) or None,
            tool_calls=tool_calls or None,
            finish_reason=finish,
            reasoning="\n".join(reasoning) or None,
            usage=Usage.from_responses(_get(response, "usage")) if response is not None else None,
        )
