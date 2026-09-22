"""DeepSeek provider profile.

V4 defaults to thinking ON when ``extra_body.thinking`` is unset, and then
requires ``reasoning_content`` to be echoed back on later turns (HTTP 400 after
the first tool call otherwise). So ``thinking`` is always set explicitly, and
effort maps onto DeepSeek's ``reasoning_effort``. V3 models are left untouched.
"""

from typing import Any

from ...reasoning_effort import DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES, clamp_effort
from .. import register_provider
from ..base import ProviderProfile


class DeepSeekProfile(ProviderProfile):
    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                model: str | None = None, **ctx: Any):
        m = (model or "").strip().lower()
        if not m.startswith("deepseek-v") or m.startswith("deepseek-v3"):     # v4+ only
            return {}, {}
        rc = reasoning_config if isinstance(reasoning_config, dict) else None
        if rc is not None and rc.get("enabled") is False:
            return {"thinking": {"type": "disabled"}}, {}
        top_level: dict[str, Any] = {}
        effort = (rc.get("effort") or "").strip().lower() if rc is not None else ""
        if effort and effort != "none":
            clamped = clamp_effort(effort, DEEPSEEK_V4_EFFORTS, DEEPSEEK_V4_OVERRIDES)
            if clamped in DEEPSEEK_V4_EFFORTS:
                top_level["reasoning_effort"] = clamped
        return {"thinking": {"type": "enabled"}}, top_level


register_provider(DeepSeekProfile(
    name="deepseek", aliases=("deepseek-chat",), display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API", signup_url="https://platform.deepseek.com/",
    env_vars=("DEEPSEEK_API_KEY",), base_url="https://api.deepseek.com/v1",
    default_model="deepseek-chat",
    fallback_models=("deepseek-v4-pro", "deepseek-v4-flash"),
    default_aux_model="deepseek-v4-flash",
))
