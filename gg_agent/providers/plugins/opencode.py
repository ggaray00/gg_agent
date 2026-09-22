"""OpenCode provider profiles: Zen, Go, and the keyless Free tier on the Zen relay.

The relays route to many upstreams; these profiles carry the per-model
reasoning translations (GLM-5.2, Kimi K2, DeepSeek, Ox Alpha).
"""

from typing import Any

from ... import reasoning_effort as re_
from .. import register_provider
from ..base import ProviderProfile
from ._common import ATTRIBUTION_HEADERS


def _flat_model_name(model: str | None) -> str:
    """Bare model id, tolerating aggregator prefixes."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def _is_deepseek_thinking_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    return (m.startswith("deepseek-v") and not m.startswith("deepseek-v3")) or m == "deepseek-reasoner"


def _is_glm_5_2_model(model: str | None) -> bool:
    m = _flat_model_name(model)
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


def _requested_effort(reasoning_config: dict | None) -> str | None:
    effort = re_.requested_effort(reasoning_config)
    return None if effort == "none" else effort


def _thinking_toggle_extras(reasoning_config: dict | None, efforts, overrides=None):
    """Moonshot/DeepSeek wire: extra_body.thinking XOR top-level reasoning_effort."""
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        return {"thinking": {"type": "disabled"}}, {}
    clamped = re_.clamp_effort(_requested_effort(reasoning_config), efforts, overrides)
    if clamped in efforts:
        return {}, {"reasoning_effort": clamped}
    return {"thinking": {"type": "enabled"}}, {}


def build_ox_alpha_reasoning_extras(reasoning_config: dict | None, model: str | None):
    """Ox Alpha (x-preview-f-free): low/high/max only, anything else 400s."""
    if _flat_model_name(model) != "x-preview-f-free":
        return {}, {}
    clamped = re_.clamp_effort(_requested_effort(reasoning_config), re_.OX_ALPHA_EFFORTS, re_.OX_ALPHA_OVERRIDES)
    return ({}, {"reasoning_effort": clamped}) if clamped in re_.OX_ALPHA_EFFORTS else ({}, {})


class OpenCodeZenProfile(ProviderProfile):
    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                model: str | None = None, **ctx: Any):
        return build_ox_alpha_reasoning_extras(reasoning_config, model)


class OpenCodeGoProfile(ProviderProfile):
    # The relay's default max_tokens (262144) exceeds what mimo-v2.5-pro accepts.
    _MODEL_MAX_TOKENS: dict[str, int] = {"mimo-v2.5-pro": 131072}

    def get_max_tokens(self, model: str | None) -> int | None:
        cap = self._MODEL_MAX_TOKENS.get(_flat_model_name(model))
        return self.default_max_tokens if cap is None else cap

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                model: str | None = None, **ctx: Any):
        if _is_glm_5_2_model(model):
            effort = _requested_effort(reasoning_config)
            if effort is None:
                return {}, {}
            clamped = re_.clamp_effort(effort, re_.GLM52_EFFORTS, re_.GLM52_OVERRIDES)
            return {}, {"reasoning_effort": clamped if clamped in re_.GLM52_EFFORTS else "high"}
        if _flat_model_name(model).startswith("kimi-k2"):
            if not isinstance(reasoning_config, dict):
                return {}, {}
            return _thinking_toggle_extras(reasoning_config, re_.KIMI_K2_EFFORTS)
        if _is_deepseek_thinking_model(model):
            return _thinking_toggle_extras(reasoning_config, re_.DEEPSEEK_V4_EFFORTS, re_.DEEPSEEK_V4_OVERRIDES)
        return {}, {}


class OpenCodeFreeProfile(OpenCodeZenProfile):
    """Keyless: the relay serves free models anonymously and 401s any bearer it
    doesn't recognise, so no credential is ever sent.

    The relay decides which clients its free tier serves; at the time of writing
    it answers other clients with 403 "free tier can only be used from within
    OpenCode". The profile is kept so it works again if that opens up.
    """

    def resolve_api_key(self) -> str:
        return ""

    def fetch_models(self, **kwargs):
        # The relay lists the whole Zen catalog; only the ``-free`` slugs are keyless.
        models = super().fetch_models(**kwargs)
        return None if models is None else [m for m in models if m.endswith("-free")]


register_provider(OpenCodeZenProfile(
    name="opencode-zen", aliases=("opencode", "opencode_zen", "zen"), display_name="OpenCode Zen",
    env_vars=("OPENCODE_ZEN_API_KEY",), base_url="https://opencode.ai/zen/v1",
    default_headers=dict(ATTRIBUTION_HEADERS), default_aux_model="gemini-3-flash",
))
register_provider(OpenCodeGoProfile(
    name="opencode-go", aliases=("opencode_go", "go", "opencode-go-sub"), display_name="OpenCode Go",
    env_vars=("OPENCODE_GO_API_KEY",), base_url="https://opencode.ai/zen/go/v1",
    default_headers=dict(ATTRIBUTION_HEADERS), default_aux_model="glm-5",
))
register_provider(OpenCodeFreeProfile(
    name="opencode-free", aliases=("free", "opencode_free"), display_name="OpenCode Free",
    description="OpenCode free models — keyless (the relay may restrict which clients it serves)",
    env_vars=(), base_url="https://opencode.ai/zen/v1",
    # The empty Authorization keeps the SDK's "Bearer <placeholder>" off the wire.
    default_headers={"Authorization": "", **ATTRIBUTION_HEADERS},
    default_model="deepseek-v4-flash-free", default_aux_model="deepseek-v4-flash-free",
))
