"""Helpers shared by several provider plugins."""

from __future__ import annotations

from typing import Any

from ..base import USER_AGENT

# Client attribution (what hermes sends as "Hermes Agent"): some gateways rank or
# rate-limit by it, and a WAF or two rejects the default SDK User-Agent.
ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/",
    "X-Title": "gg-agent",
    "User-Agent": USER_AGENT,
}

_HIGH_EFFORTS = {"high", "xhigh", "max", "ultra"}


def build_gemini_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    """reasoning_config -> Gemini ``thinkingConfig`` (camelCase, native shape).

    Gemini-only: Gemma/PaLM on the same endpoint 400 on the field, even as
    ``{"includeThoughts": False}``, so non-Gemini models get nothing.
    """
    if not isinstance(reasoning_config, dict):
        return None
    normalized = (model or "").strip().lower().removeprefix("google/")
    if not normalized.startswith("gemini"):
        return None
    effort = str(reasoning_config.get("effort", "medium") or "medium").strip().lower()
    if reasoning_config.get("enabled") is False or effort == "none":
        return {"includeThoughts": False}
    config: dict[str, Any] = {"includeThoughts": True}
    # Gemini 2.5 takes a token budget; don't guess one from a coarse effort level.
    if normalized.startswith("gemini-2.5-"):
        return config
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
        effort = "medium"
    # Gemini 3 Flash documents low/medium/high, Gemini 3 Pro only low/high.
    if normalized.startswith("gemini-3"):
        if "flash" in normalized:
            config["thinkingLevel"] = ("low" if effort in {"minimal", "low"}
                                       else "high" if effort in _HIGH_EFFORTS else "medium")
        elif "pro" in normalized:
            config["thinkingLevel"] = "high" if effort in _HIGH_EFFORTS else "low"
    return config


def snake_case_gemini_thinking_config(config: dict | None) -> dict | None:
    """Native thinkingConfig keys -> the OpenAI-compat endpoint's snake_case names."""
    if not isinstance(config, dict) or not config:
        return None
    out: dict[str, Any] = {}
    include, level, budget = config.get("includeThoughts"), config.get("thinkingLevel"), config.get("thinkingBudget")
    if isinstance(include, bool):
        out["include_thoughts"] = include
    if isinstance(level, str) and level.strip():
        out["thinking_level"] = level.strip().lower()
    if isinstance(budget, (int, float)):
        out["thinking_budget"] = int(budget)
    return out or None


def gemini_openai_compat_extra_body(model: str, reasoning_config: dict | None) -> dict[str, Any]:
    """``{"extra_body": {"google": {"thinking_config": ...}}}`` — Google's OpenAI-compat
    endpoint reads vendor fields from a literal nested ``extra_body`` key."""
    thinking = snake_case_gemini_thinking_config(build_gemini_thinking_config(model, reasoning_config))
    return {"extra_body": {"google": {"thinking_config": thinking}}} if thinking else {}
