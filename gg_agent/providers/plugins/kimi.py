"""Kimi / Moonshot provider profiles (international + China)."""

from typing import Any
from urllib.parse import urlparse

from ...reasoning_effort import KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES, clamp_effort, requested_effort
from .. import register_provider
from ..base import OMIT_TEMPERATURE, ProviderProfile
from ._common import ATTRIBUTION_HEADERS


def _is_confirmed_kimi_coding_url(base_url: str) -> bool:
    """True only for Kimi Code's canonical HTTPS API surfaces."""
    try:
        p = urlparse(base_url)
        port = p.port
    except ValueError:
        return False
    return (p.scheme.lower() == "https" and (p.hostname or "").lower() == "api.kimi.com"
            and port in (None, 443) and p.username is None and p.password is None
            and p.path.rstrip("/") in {"/coding", "/coding/v1"} and not p.query and not p.fragment)


class KimiProfile(ProviderProfile):
    """Temperature omitted (server-managed); thinking XOR reasoning_effort."""

    def resolve_base_url(self) -> str:
        # sk-kimi-* keys belong to Kimi Code, which lives on its own host.
        if self.resolve_api_key().startswith("sk-kimi-"):
            return "https://api.kimi.com/coding/v1"
        return super().resolve_base_url()

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        effective = (base_url or self.base_url or "").rstrip("/")
        confirmed = _is_confirmed_kimi_coding_url(effective)
        if confirmed and urlparse(effective).path.rstrip("/") == "/coding":
            effective += "/v1"
        models = super().fetch_models(api_key=api_key, base_url=effective or None, timeout=timeout)
        if models is None or confirmed:
            return models
        # The bare ``k3`` slug is only served on Kimi Code.
        return [m for m in models if m.strip().lower() != "k3"]

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None, **ctx: Any):
        # Moonshot 400s when both extra_body.thinking and reasoning_effort are sent.
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled", True) is False:
            return {"thinking": {"type": "disabled"}}, {}
        effort = requested_effort(reasoning_config)
        k3_effort = clamp_effort(effort, KIMI_K3_EFFORTS, KIMI_K3_OVERRIDES) if effort != "none" else None
        if k3_effort in KIMI_K3_EFFORTS:
            return {}, {"reasoning_effort": k3_effort}
        return {"thinking": {"type": "enabled"}}, {}


def _kimi(name: str, aliases: tuple, env_vars: tuple, base_url: str, display: str) -> KimiProfile:
    return KimiProfile(
        name=name, aliases=aliases, display_name=display, env_vars=env_vars, base_url=base_url,
        fixed_temperature=OMIT_TEMPERATURE, default_max_tokens=32000,
        default_headers=dict(ATTRIBUTION_HEADERS), default_aux_model="kimi-k2-turbo-preview",
    )


register_provider(_kimi("kimi-coding", ("kimi", "moonshot", "kimi-for-coding"),
                        ("KIMI_API_KEY", "KIMI_CODING_API_KEY"), "https://api.moonshot.ai/v1", "Kimi / Moonshot"))
register_provider(_kimi("kimi-coding-cn", ("kimi-cn", "moonshot-cn"), ("KIMI_CN_API_KEY",),
                        "https://api.moonshot.cn/v1", "Kimi / Moonshot (China)"))
