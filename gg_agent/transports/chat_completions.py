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
from .types import NormalizedResponse, ToolCall, Usage


class ChatCompletionsTransport(ProviderTransport):

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def build_client(self, *, api_key: str, base_url: str, profile: Any) -> Any:
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=api_key or "not-needed",     # local servers accept anything
            base_url=base_url or None,
            default_headers=dict(getattr(profile, "default_headers", {}) or {}) or None,
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
        profile = params.pop("profile", None)
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

        max_tokens = params.pop("max_tokens", None) or getattr(profile, "default_max_tokens", None)
        if max_tokens:
            api_kwargs["max_tokens"] = max_tokens

        extra_body = dict(params.pop("extra_body", None) or {})
        if profile is not None:
            extra_body.update(profile.build_extra_body(model=model))
        if extra_body:
            api_kwargs["extra_body"] = extra_body

        api_kwargs.update({k: v for k, v in params.items() if v is not None})
        return api_kwargs

    async def call(self, client: Any, **api_kwargs) -> Any:
        return await client.chat.completions.create(**api_kwargs)

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
