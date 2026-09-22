"""OpenRouter provider profile: sticky routing, provider preferences, reasoning passthrough."""

from __future__ import annotations

import logging
from typing import Any

from .. import register_provider
from ..base import ProviderProfile
from ._common import ATTRIBUTION_HEADERS

logger = logging.getLogger(__name__)

_CACHE: list[str] | None = None

# Anthropic models that still accept an explicit "disable thinking". Claude 4.6+
# mandate reasoning and 400 on any disable form, so UNKNOWN Anthropic models
# default to "cannot disable".
_ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS = (
    "claude-3",
    "claude-opus-4-0", "claude-opus-4.0", "claude-opus-4-1", "claude-opus-4.1",
    "claude-sonnet-4-0", "claude-sonnet-4.0",
    "claude-opus-4-2025", "claude-sonnet-4-2025",
    "claude-opus-4-5", "claude-opus-4.5",
    "claude-sonnet-4-5", "claude-sonnet-4.5",
    "claude-haiku-4-5", "claude-haiku-4.5",
)


def anthropic_reasoning_is_mandatory(model: str | None) -> bool:
    m = (model or "").lower()
    if not m.startswith(("anthropic/", "claude")) and "claude" not in m:
        return False
    return not any(sub in m for sub in _ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS)


# OpenAI speed tiers are ENDPOINTS of the base model on OpenRouter (tags
# ``openai/fast``, ``openai/flex``), and an unknown ``-fast`` suffix silently routes
# to the standard tier. So a tier slug becomes the base slug + ``provider.only``.
_SPEED_TIER_ENDPOINTS = {"": ("openai", "azure", "azure/us"), "-fast": ("openai/fast",), "-flex": ("openai/flex",)}
_SPEED_TIERED_BASES = ("openai/gpt-6-astra", "openai/gpt-6-astra-pro")
OPENROUTER_ENDPOINT_PINS: dict[str, tuple[str, tuple[str, ...]]] = {
    base + suffix: (base, tags) for base in _SPEED_TIERED_BASES for suffix, tags in _SPEED_TIER_ENDPOINTS.items()
}


class OpenRouterProfile(ProviderProfile):
    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        """Public catalog (no auth), cached per process."""
        global _CACHE  # noqa: PLW0603
        if _CACHE is None:
            _CACHE = super().fetch_models(api_key=None, base_url=base_url, timeout=timeout)
        return _CACHE

    def build_extra_body(self, *, session_id: str | None = None, **ctx: Any) -> dict[str, Any]:
        body: dict[str, Any] = {}
        # Top-level session_id is OpenRouter's sticky routing key: requests of one
        # session land on the same upstream, so its prompt cache stays warm.
        if session_id:
            body["session_id"] = session_id
        prefs = ctx.get("provider_preferences")
        model = ctx.get("model") or ""
        pin = OPENROUTER_ENDPOINT_PINS.get(model)
        # The tier pin owns ``only`` — except on the base slug, where an explicit
        # user ``only`` is the stronger intent.
        if pin and not (pin[0] == model and (prefs or {}).get("only")):
            prefs = {**(prefs or {}), "only": list(pin[1])}
        if prefs:
            body["provider"] = prefs
        score = ctx.get("openrouter_min_coding_score")
        if model == "openrouter/pareto-code" and score not in (None, ""):
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = None
            if score_f is not None and 0.0 <= score_f <= 1.0:
                body["plugins"] = [{"id": "pareto-router", "min_coding_score": score_f}]
        return body

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None, supports_reasoning: bool = False,
                                model: str | None = None, session_id: str | None = None, **ctx: Any):
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        pin = OPENROUTER_ENDPOINT_PINS.get(model or "")
        if pin and pin[0] != model:
            top_level["model"] = pin[0]
        if supports_reasoning:
            if anthropic_reasoning_is_mandatory(model):
                # Adaptive-thinking Claude: any ``reasoning`` field makes OpenRouter
                # emit ``thinking: disabled`` on tool-continuation turns -> 400. The
                # effort still reaches Anthropic through top-level ``verbosity``.
                cfg = reasoning_config or {}
                effort = cfg.get("effort")
                if cfg.get("enabled", True) is not False and effort and effort != "none":
                    top_level["verbosity"] = effort
            elif reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        # xAI pins its prompt cache per backend server via this header.
        if session_id and model and model.startswith(("x-ai/grok-", "xai/grok-")):
            top_level["extra_headers"] = {"x-grok-conv-id": session_id}
        return extra_body, top_level


register_provider(OpenRouterProfile(
    name="openrouter", aliases=("or",), display_name="OpenRouter",
    description="OpenRouter — unified API for 200+ models", signup_url="https://openrouter.ai/keys",
    env_vars=("OPENROUTER_API_KEY",), base_url="https://openrouter.ai/api/v1",
    models_url="https://openrouter.ai/api/v1/models",
    default_model="anthropic/claude-sonnet-5",
    fallback_models=(
        "anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "deepseek/deepseek-chat", "google/gemini-3.8-flash",
        "google/gemini-3.7-flash", "qwen/qwen3-plus",
    ),
    default_headers={k: v for k, v in ATTRIBUTION_HEADERS.items() if k != "User-Agent"},
))
