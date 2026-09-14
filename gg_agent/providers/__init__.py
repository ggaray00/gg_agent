"""Provider registry — the single source of truth for "which LLM are we talking to".

Every other layer (client construction, model listing, transport selection) reads
from a ``ProviderProfile`` here instead of keeping parallel data.

In hermes-agent the profiles live as plugins under ``plugins/model-providers/<name>/``
and are lazily discovered. Here they are declared inline; the registry API is the
same shape, so swapping in a plugin loader later touches only this file.

Mirrors hermes-agent: providers/__init__.py
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from .base import OMIT_TEMPERATURE, ProviderProfile
from .copilot_auth import (
    COPILOT_DEFAULT_BASE_URL,
    copilot_request_headers,
    exchange_copilot_token,
    get_copilot_credentials,
)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}


def register_provider(profile: ProviderProfile) -> ProviderProfile:
    """Add (or replace) a profile. Aliases resolve to the canonical name."""
    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name
    return profile


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up by canonical name or alias; None when unknown."""
    if not name:
        return None
    key = name.strip().lower()
    return _REGISTRY.get(_ALIASES.get(key, key))


def list_providers() -> list[ProviderProfile]:
    return sorted(_REGISTRY.values(), key=lambda p: p.name)


def iter_configured() -> Iterator[ProviderProfile]:
    """Profiles whose credentials are actually present in the environment."""
    for profile in list_providers():
        if profile.has_credentials():
            yield profile


# ── Built-in profiles ────────────────────────────────────────────────────────
# Anything OpenAI-compatible only needs a base_url + env var.


def _token_kind(token: str) -> str:
    """Token TYPE for display — never any of the secret itself."""
    for prefix in ("ghu_", "gho_", "ghp_", "github_pat_", "ghs_"):
        if token.startswith(prefix):
            return f"{prefix}…"
    return "opaque token"


class CopilotProfile(ProviderProfile):
    """GitHub Copilot — an OAuth token exchanged for a short-lived API token.

    This is the profile hook system earning its keep: Copilot needs a live
    credential *and* an account-specific base URL resolved per turn, and none of
    that leaks into the Agent, the loop or the transport.
    """

    def resolve_api_key(self) -> str:
        try:
            return get_copilot_credentials()[0]
        except Exception:
            return ""

    def has_credentials(self) -> bool:
        """Auto-detect on a COPILOT-SPECIFIC signal only, and never over the network.

        ``GH_TOKEN`` / ``GITHUB_TOKEN`` are usually exported for `gh` or CI, not for
        Copilot — auto-selecting this provider off one of them would hijack every run
        on a machine that has a perfectly good OPENAI_API_KEY. Both still work when
        Copilot is asked for explicitly (``-p copilot``).
        """
        import os
        from pathlib import Path

        from .copilot_auth import _CRED_FILES
        if os.getenv("COPILOT_GITHUB_TOKEN", "").strip():
            return True
        return any(Path(os.path.expanduser(f)).is_file() for f in _CRED_FILES)

    def resolve_credentials(self) -> tuple[str, str]:
        # Re-read every turn: the exchanged token expires in ~30 minutes, and
        # enterprise accounts are not on the public base URL.
        api_token, base_url = get_copilot_credentials()
        return api_token, base_url or self.base_url

    def credential_status(self) -> str:
        from .copilot_auth import resolve_github_token
        token, source = resolve_github_token()
        if not token:
            return "no GitHub token found (run `gg-agent login`)"
        try:
            api_token, expires_at, base_url = exchange_copilot_token(token)
        except Exception as exc:
            return f"GitHub token from {source} ({_token_kind(token)}) — exchange failed: {exc}"
        return (f"GitHub token from {source} ({_token_kind(token)}) → Copilot token valid for "
                f"{max(int(expires_at - time.time()), 0)}s @ {base_url}")


class _OpenRouterProfile(ProviderProfile):
    def build_extra_body(self, **ctx):
        # OpenRouter-specific routing preferences ride in extra_body.
        return {"provider": {"require_parameters": True}}


register_provider(ProviderProfile(
    name="openai",
    display_name="OpenAI",
    env_vars=("OPENAI_API_KEY",),
    base_url="https://api.openai.com/v1",
    default_model="gpt-4.1",
    fallback_models=("gpt-4.1", "gpt-4.1-mini", "o4-mini"),
    default_aux_model="gpt-4.1-mini",
))

register_provider(ProviderProfile(
    name="anthropic",
    display_name="Anthropic",
    api_mode="anthropic_messages",           # <- different transport, same loop
    env_vars=("ANTHROPIC_API_KEY",),
    base_url="https://api.anthropic.com",
    default_model="claude-sonnet-5",
    fallback_models=("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"),
    default_max_tokens=8192,                 # Anthropic requires max_tokens
    default_aux_model="claude-haiku-4-5-20251001",
))

register_provider(_OpenRouterProfile(
    name="openrouter",
    display_name="OpenRouter",
    env_vars=("OPENROUTER_API_KEY",),
    base_url="https://openrouter.ai/api/v1",
    default_model="anthropic/claude-sonnet-5",
    default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "gg-agent"},
))

register_provider(ProviderProfile(
    name="groq",
    display_name="Groq",
    env_vars=("GROQ_API_KEY",),
    base_url="https://api.groq.com/openai/v1",
    default_model="llama-3.3-70b-versatile",
))

register_provider(ProviderProfile(
    name="deepseek",
    display_name="DeepSeek",
    env_vars=("DEEPSEEK_API_KEY",),
    base_url="https://api.deepseek.com/v1",
    default_model="deepseek-chat",
))

register_provider(ProviderProfile(
    name="ollama",
    display_name="Ollama (local)",
    env_vars=(),                             # no key needed
    base_url="http://localhost:11434/v1",
    default_model="qwen2.5-coder:7b",
    fixed_temperature=0.0,
))

register_provider(CopilotProfile(
    name="copilot",
    display_name="GitHub Copilot",
    aliases=("github-copilot", "github", "gh-copilot"),
    # Copilot speaks the OpenAI Chat Completions shape, so it reuses that transport
    # unchanged — only auth and headers are special.
    api_mode="chat_completions",
    env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
    base_url=COPILOT_DEFAULT_BASE_URL,
    default_model="gpt-4.1",
    fallback_models=("gpt-4.1", "gpt-5", "claude-sonnet-4.5", "o4-mini"),
    default_headers=copilot_request_headers(),
    default_aux_model="gpt-4.1-mini",
))

__all__ = [
    "OMIT_TEMPERATURE", "ProviderProfile", "CopilotProfile", "register_provider",
    "get_provider_profile", "list_providers", "iter_configured",
]
