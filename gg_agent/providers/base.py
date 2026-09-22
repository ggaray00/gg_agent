"""Provider profile base class.

A ProviderProfile declares everything about an inference provider in one place:
auth, endpoints, client quirks, request-time quirks. The transport reads this
instead of receiving a dozen boolean flags.

Profiles are DECLARATIVE — they describe the provider's behaviour. They do NOT
own client construction, credential rotation or streaming; those stay on Agent.
The hooks below are the escape hatches for providers whose quirks don't fit in
a field (reasoning knobs, message rewrites, non-HTTP clients).

Mirrors hermes-agent: providers/base.py
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Sentinel for "omit temperature entirely" (some providers manage it server-side).
OMIT_TEMPERATURE = object()

USER_AGENT = "gg-agent/0.1"


def is_base_url_var(var: str) -> bool:
    """``env_vars`` mixes keys and endpoint overrides (hermes convention); the
    ``*_BASE_URL`` ones are never a credential."""
    return var.upper().endswith("_BASE_URL")


@dataclass
class ProviderProfile:
    """One inference provider, declared once."""

    # ── Identity ──────────────────────────────────────────────
    name: str
    api_mode: str = "chat_completions"   # which transport owns the data path
    aliases: tuple[str, ...] = ()
    display_name: str = ""
    description: str = ""                # one-liner for listings
    signup_url: str = ""                 # where to get a key

    # ── Auth & endpoints ──────────────────────────────────────
    # Checked in order; the first non-empty key wins. Vars ending in ``_BASE_URL``
    # are endpoint overrides, not keys (see ``resolve_base_url``).
    env_vars: tuple[str, ...] = ()
    base_url: str = ""
    models_url: str = ""                 # explicit catalog endpoint; default {base_url}/models
    # api_key | oauth_device_code | oauth_external | copilot | aws_sdk | vertex | external_process
    auth_type: str = "api_key"
    # Anthropic-Messages providers that want ``Authorization: Bearer`` instead of x-api-key.
    bearer_auth: bool = False
    supports_health_check: bool = True   # False = /models 401s even with a valid key

    # ── Capabilities ──────────────────────────────────────────
    supports_vision: bool = False
    # False for providers that take images on user turns but 400 on list-type tool content.
    supports_vision_tool_messages: bool = True
    # Opt-in: many OpenAI-compatible endpoints 400 on unknown top-level fields.
    supports_prompt_cache_key: bool = False

    # ── External-process providers (auth_type="external_process") ──
    process_command: str = ""
    process_args: tuple[str, ...] = ()
    process_command_env_vars: tuple[str, ...] = ()   # env overrides for the binary, in order
    process_args_env_var: str = ""                   # env override for argv (shlex-split)

    # ── Model catalog ─────────────────────────────────────────
    default_model: str = ""              # empty = first of fallback_models
    fallback_models: tuple[str, ...] = ()
    hostname: str = ""                   # derived from base_url when empty
    # Context window, when the provider pins one regardless of model name.
    # 0 = let the model name decide (see ``compression.MODEL_CONTEXT_LENGTHS``).
    context_length: int = 0

    # ── Request-level quirks ──────────────────────────────────
    default_headers: dict[str, str] = field(default_factory=dict)
    fixed_temperature: Any = None        # None = caller's default, OMIT_TEMPERATURE = don't send
    default_max_tokens: int | None = None
    default_aux_model: str = ""          # cheap model for summaries/compression

    def __post_init__(self) -> None:
        if not self.default_model and self.fallback_models:
            self.default_model = self.fallback_models[0]

    # ── Credentials ───────────────────────────────────────────

    @property
    def key_env_vars(self) -> tuple[str, ...]:
        return tuple(v for v in self.env_vars if not is_base_url_var(v))

    def resolve_api_key(self) -> str:
        """First non-empty key among ``env_vars``, or "" (e.g. local Ollama)."""
        for var in self.key_env_vars:
            value = os.getenv(var, "").strip()
            if value:
                return value
        return ""

    def resolve_base_url(self) -> str:
        """A ``*_BASE_URL`` env override when set, else the declared base_url."""
        for var in self.env_vars:
            if is_base_url_var(var):
                value = os.getenv(var, "").strip()
                if value:
                    return value.rstrip("/")
        return self.base_url

    def resolve_credentials(self) -> tuple[str, str]:
        """LIVE ``(api_key, base_url)`` for the next request.

        Called once at Agent construction and again before every turn, so a
        provider whose credential is short-lived (OAuth exchange, rotating JWT)
        can hand back a fresh one without the caller knowing.
        """
        return self.resolve_api_key(), self.resolve_base_url()

    def has_credentials(self) -> bool:
        """Cheap "is this provider usable?" probe for auto-detection.

        MUST NOT hit the network — it runs for every provider on every cold start.
        """
        return bool(self.resolve_api_key()) if self.key_env_vars else True

    def credential_status(self) -> str:
        """One-line human description of where the credential came from."""
        for var in self.key_env_vars:
            if os.getenv(var, "").strip():
                return f"credential present via {var}"
        if self.key_env_vars:
            return f"no credential (set {self.key_env_vars[0]})"
        return "no credential needed"

    # ── Hooks (override in a subclass for complex providers) ──

    def get_hostname(self) -> str:
        if self.hostname:
            return self.hostname
        return urlparse(self.base_url).hostname or "" if self.base_url else ""

    def resolve_aux_model(self, *, vision: bool = False) -> str:
        """LIVE cheap-model id for auxiliary tasks, or "" to use ``default_aux_model``.
        Must be cheap (cache it) and never raise."""
        return ""

    def default_vision_model(self) -> str | None:
        return None

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing. Default: pass through."""
        return messages

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Provider-specific ``extra_body`` fields (routing prefs, thinking config...)."""
        return {}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(extra_body_additions, top_level_kwargs)``.

        The split exists because providers disagree on where reasoning goes:
        OpenRouter wants ``extra_body.reasoning``, Kimi a top-level
        ``reasoning_effort``. ``context`` carries model, base_url, session_id and
        supports_reasoning; accept ``**context`` so new keys never break you.
        """
        return {}, {}

    def build_responses_extras(self, **context: Any) -> dict[str, Any]:
        """Top-level kwargs merged into a Responses API request (api_mode
        "codex_responses"): extra headers, cache keys... ``context`` as above."""
        return {}

    def get_max_tokens(self, model: str | None) -> int | None:
        """Default output cap for *model* when the caller set none."""
        return self.default_max_tokens

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        """Declared effort vocabulary: None = unknown, () = no reasoning fields at all,
        else clamp onto these. Hot path: answer from a cache, never block."""
        return None

    def create_client(self, **client_kwargs: Any) -> Any | None:
        """A provider-specific client, or None for the transport's standard one.
        Used by providers whose wire isn't HTTP at all (ACP subprocesses)."""
        return None

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None,
                     timeout: float = 8.0) -> list[str] | None:
        """Live model ids from the catalog endpoint, or None when unavailable.

        A caller ``base_url`` that differs from the declared one is a custom
        endpoint and wins over ``models_url``.
        """
        caller_base = (base_url or "").strip().rstrip("/")
        if caller_base and caller_base != (self.base_url or "").rstrip("/"):
            url = caller_base + "/models"
        else:
            url = self.models_url or (self.base_url.rstrip("/") + "/models" if self.base_url else "")
        if not url:
            return None
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        # Some catalogs sit behind a WAF that 403s the default Python-urllib UA.
        req.add_header("User-Agent", USER_AGENT)
        for k, v in self.default_headers.items():
            if v:
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None
        items = data if isinstance(data, list) else data.get("data", [])
        return [m["id"] for m in items if isinstance(m, dict) and "id" in m]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.display_name or self.name
