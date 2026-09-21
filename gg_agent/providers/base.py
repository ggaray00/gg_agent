"""Provider profile base class.

A ProviderProfile declares everything about an inference provider in one place:
auth, endpoints, request-time quirks. The transport reads this instead of
receiving a dozen boolean flags.

Profiles are DECLARATIVE — they describe the provider's behaviour. They do NOT
own client construction, credential rotation or streaming; those stay on Agent.

Mirrors hermes-agent: providers/base.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Sentinel for "omit temperature entirely" (some providers manage it server-side).
OMIT_TEMPERATURE = object()


@dataclass
class ProviderProfile:
    """One inference provider, declared once."""

    # ── Identity ──────────────────────────────────────────────
    name: str
    api_mode: str = "chat_completions"   # which transport owns the data path
    aliases: tuple[str, ...] = ()
    display_name: str = ""

    # ── Auth & endpoints ──────────────────────────────────────
    env_vars: tuple[str, ...] = ()       # checked in order; first non-empty wins
    base_url: str = ""

    # ── Model catalog ─────────────────────────────────────────
    default_model: str = ""
    fallback_models: tuple[str, ...] = ()
    # Context window, when the provider pins one regardless of model name.
    # 0 = let the model name decide (see ``compression.MODEL_CONTEXT_LENGTHS``).
    context_length: int = 0

    # ── Request-level quirks ──────────────────────────────────
    default_headers: dict[str, str] = field(default_factory=dict)
    fixed_temperature: Any = None        # None = caller's default, OMIT_TEMPERATURE = don't send
    default_max_tokens: int | None = None
    default_aux_model: str = ""          # cheap model for summaries/compression

    # ── Hooks (override in a subclass for complex providers) ──

    def resolve_api_key(self) -> str:
        """First non-empty value among ``env_vars``, or "" (e.g. local Ollama)."""
        for var in self.env_vars:
            value = os.getenv(var, "").strip()
            if value:
                return value
        return ""

    def resolve_credentials(self) -> tuple[str, str]:
        """LIVE ``(api_key, base_url)`` for the next request.

        Called once at Agent construction and again before every turn, so a
        provider whose credential is short-lived (OAuth exchange, rotating JWT)
        can hand back a fresh one without the caller knowing. Default: the
        static env key and the declared base_url.
        """
        return self.resolve_api_key(), self.base_url

    def has_credentials(self) -> bool:
        """Cheap "is this provider usable?" probe for auto-detection.

        MUST NOT hit the network — it runs for every provider on every cold start.
        """
        return bool(self.resolve_api_key()) if self.env_vars else True

    def credential_status(self) -> str:
        """One-line human description of where the credential came from."""
        key, _ = self.resolve_credentials()
        if not key:
            return "no credential" if self.env_vars else "no credential needed"
        return f"credential present via {self.env_vars[0] if self.env_vars else 'config'}"

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing. Default: pass through."""
        return messages

    def build_extra_body(self, **ctx: Any) -> dict[str, Any]:
        """Provider-specific ``extra_body`` (routing prefs, thinking config...)."""
        return {}

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.display_name or self.name
