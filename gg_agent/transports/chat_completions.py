"""OpenAI Chat Completions transport (api_mode="chat_completions").

Covers OpenAI plus every OpenAI-compatible endpoint (OpenRouter, Groq,
DeepSeek, Ollama, vLLM...). This is the "native" format of the loop, so
conversion is mostly a pass-through with sanitisation.

Mirrors hermes-agent: agent/transports/chat_completions.py
"""

from __future__ import annotations

from typing import Any

from ..providers.base import OMIT_TEMPERATURE
from .base import ProviderTransport
from .streaming import StreamDropped, StreamHooks, StreamInterrupted, ToolCallAccumulator, aclose_quietly
from .types import NormalizedResponse, ToolCall, Usage


class ChatCompletionsTransport(ProviderTransport):
    supports_streaming = True

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def build_client(self, *, api_key: str, base_url: str, profile: Any) -> Any:
        headers = dict(getattr(profile, "default_headers", {}) or {})
        # A provider whose wire isn't HTTP (an ACP subprocess) brings its own client.
        if profile is not None:
            custom = profile.create_client(api_key=api_key, base_url=base_url, default_headers=headers)
            if custom is not None:
                return custom
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=api_key or "not-needed",     # local servers accept anything
            base_url=base_url or None,
            default_headers=headers or None,
            max_retries=0,                        # the loop owns retries
        )

    # ── Conversion ───────────────────────────────────────────────────────

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """Drop internal bookkeeping keys the API rejects; keep the wire shape."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if not k.startswith("_")}
            # An assistant turn with tool_calls may carry content=None; the API wants "".
            if clean.get("role") == "assistant" and clean.get("content") is None:
                clean["content"] = ""
            out.append(clean)
        return out

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Registry schemas ({name, description, parameters}) -> OpenAI function tools."""
        return [{"type": "function", "function": t} for t in tools or []]

    def build_kwargs(self, model: str, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None, **params) -> dict[str, Any]:
        """Every provider quirk comes from the profile's fields and hooks.

        ``params`` beyond the SDK's own: ``profile``, ``reasoning_config``
        ({enabled, effort} or None = provider default), ``session_id`` (sticky
        routing key) and ``base_url`` (the live endpoint, for hooks that care).
        """
        profile = params.pop("profile", None)
        # Swallowed, not forwarded: every OpenAI-compatible endpoint caches
        # prefixes on its own with no request parameter, so there is nothing to
        # ask for — and an unknown key here would be a 400.
        params.pop("cache_prompt", None)
        reasoning_config = params.pop("reasoning_config", None)
        session_id = params.pop("session_id", None)
        base_url = params.pop("base_url", None)
        sanitized = self.convert_messages(messages)
        if profile is not None:
            sanitized = profile.prepare_messages(sanitized)

        api_kwargs: dict[str, Any] = {"model": model, "messages": sanitized}
        if tools:
            api_kwargs["tools"] = self.convert_tools(tools)
            api_kwargs["tool_choice"] = params.pop("tool_choice", "auto")

        temperature = params.pop("temperature", None)
        if profile is not None and profile.fixed_temperature is not None:
            temperature = profile.fixed_temperature
        if temperature is not None and temperature is not OMIT_TEMPERATURE:
            api_kwargs["temperature"] = temperature

        max_tokens = params.pop("max_tokens", None) or (profile.get_max_tokens(model) if profile else None)
        if max_tokens:
            api_kwargs["max_tokens"] = max_tokens

        extra_body = dict(params.pop("extra_body", None) or {})
        if profile is not None:
            context = {"model": model, "base_url": base_url, "session_id": session_id,
                       "reasoning_config": reasoning_config}
            extra_body.update(profile.build_extra_body(**context))
            # Reasoning fields only go out when the caller asked for reasoning:
            # without a model catalog, "supported" is the user's say-so.
            body_extras, top_level = profile.build_api_kwargs_extras(
                supports_reasoning=reasoning_config is not None, **context)
            extra_body.update(body_extras)
            api_kwargs.update(top_level)
        if extra_body:
            api_kwargs["extra_body"] = extra_body

        api_kwargs.update({k: v for k, v in params.items() if v is not None})
        return api_kwargs

    async def call(self, client: Any, **api_kwargs) -> Any:
        return await client.chat.completions.create(**api_kwargs)

    async def call_stream(self, client: Any, hooks: StreamHooks, **api_kwargs) -> NormalizedResponse:
        stream = await client.chat.completions.create(
            **api_kwargs, stream=True, stream_options={"include_usage": True})

        # Some endpoints ignore stream=True and answer in one piece. Take it, and
        # replay it through the hooks so the caller still sees the text.
        if not hasattr(stream, "__aiter__"):
            response = self.normalize_response(stream)
            if response.reasoning:
                hooks.on_reasoning(response.reasoning)
            if response.content:
                hooks.on_text(response.content)
            for tc in response.tool_calls or []:
                hooks.on_tool_start(tc.name)
            return response

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tools = ToolCallAccumulator()
        finish_reason: str | None = None
        usage: Any = None
        try:
            async for chunk in stream:
                if hooks.is_interrupted():
                    raise StreamInterrupted()
                # Usage arrives on a final chunk with no choices: read it first.
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    hooks.on_reasoning(reasoning)

                text = getattr(delta, "content", None)
                if text:
                    content_parts.append(text)
                    # Once a tool call has started, trailing text is not the answer.
                    if not tools:
                        hooks.on_text(text)

                for tc_delta in getattr(delta, "tool_calls", None) or []:
                    name = tools.feed(tc_delta)
                    if name:
                        hooks.on_tool_start(name)
        finally:
            await aclose_quietly(stream)

        tool_calls, truncated = tools.materialize()
        if truncated and finish_reason is None:
            raise StreamDropped("stream connection dropped mid tool-call")
        if finish_reason is None:
            finish_reason = "tool_calls" if tool_calls else "stop"
        return NormalizedResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls or None,
            finish_reason=self.map_finish_reason(finish_reason),
            reasoning="".join(reasoning_parts) or None,
            usage=Usage.from_openai(usage) if usage is not None else None,
        )

    # ── Normalization ────────────────────────────────────────────────────

    def normalize_response(self, response: Any) -> NormalizedResponse:
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (message.tool_calls or [])
        ] or None
        return NormalizedResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=self.map_finish_reason(choice.finish_reason),
            # Several providers put chain-of-thought in a non-standard field.
            reasoning=getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None),
            usage=Usage.from_openai(getattr(response, "usage", None)),
        )
