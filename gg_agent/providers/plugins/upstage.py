"""Upstage Solar provider profile: top-level ``reasoning_effort`` (low|medium|high).

Solar's server default is ``minimal`` (reasoning off) — wrong for agentic work —
so an unset reasoning_config turns reasoning ON at ``medium``. Explicit settings win.
"""

from typing import Any

from ...reasoning_effort import EFFORT_LADDER, SOLAR_EFFORTS, clamp_effort
from .. import register_provider
from ..base import ProviderProfile

# Deny-list: new Solar models are assumed reasoning-capable.
_NON_REASONING_MODEL_MARKERS = ("solar-mini", "syn-pro")


class UpstageProfile(ProviderProfile):
    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                model: str | None = None, **ctx: Any):
        m = (model or "").strip().lower()
        if any(marker in m for marker in _NON_REASONING_MODEL_MARKERS):
            return {}, {}
        if not isinstance(reasoning_config, dict):
            return {}, {"reasoning_effort": "medium"}
        if reasoning_config.get("enabled") is False:
            return {}, {}                   # Solar's own default: minimal = off
        effort = (reasoning_config.get("effort") or "").strip().lower()
        if not effort:
            return {}, {"reasoning_effort": "medium"}
        if effort == "minimal":
            return {}, {}
        mapped = clamp_effort(effort, SOLAR_EFFORTS)
        if mapped not in SOLAR_EFFORTS:
            # A bespoke level outside the ladder runs at full strength.
            mapped = "high" if effort not in EFFORT_LADDER else None
        return {}, ({"reasoning_effort": mapped} if mapped else {})


register_provider(UpstageProfile(
    name="upstage", aliases=("solar",), display_name="Upstage Solar", description="Upstage (Solar API)",
    signup_url="https://console.upstage.ai/api-keys",
    env_vars=("UPSTAGE_API_KEY", "UPSTAGE_BASE_URL"), base_url="https://api.upstage.ai/v1",
    fallback_models=("solar-pro3",),
))
