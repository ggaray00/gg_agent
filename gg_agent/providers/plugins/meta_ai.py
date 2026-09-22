"""Meta Model API (Muse Spark) provider profile — https://api.meta.ai/v1.

Runs on the Responses API: Muse prompt caching only engages there (0 cached
tokens on chat/completions vs 93-99% hits on /v1/responses).
"""

from typing import Any

from ...reasoning_effort import META_AI_EFFORTS, clamp_effort
from .. import register_provider
from ..base import ProviderProfile


class MetaAIProfile(ProviderProfile):
    # The live catalog also lists image-generation and voice models.
    _NON_CHAT_PREFIXES = ("muse-image-", "muse-voice-")

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        live = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if live is None:
            return None
        return [m for m in live if not m.startswith(self._NON_CHAT_PREFIXES)]

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        return META_AI_EFFORTS          # the Responses path clamps onto this ("none" 400s)

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None, **ctx: Any):
        """Muse always accepts reasoning_effort but 400s on ``none``: disabled ->
        ``minimal`` (closest to off); unset/bespoke -> ``medium``."""
        rc = reasoning_config or {}
        effort = str(rc.get("effort") or "").strip().lower()
        if rc.get("enabled") is False or effort == "none":
            mapped = "minimal"
        else:
            clamped = clamp_effort(effort, META_AI_EFFORTS)
            mapped = clamped if clamped in META_AI_EFFORTS else "medium"
        return {}, {"reasoning_effort": mapped}


register_provider(MetaAIProfile(
    name="meta-ai", aliases=("meta", "muse", "muse-spark", "model-api", "msl"), display_name="Meta Model API",
    description="Meta Muse Spark family (Meta Superintelligence Labs)",
    signup_url="https://developer.meta.com/ai/",
    env_vars=("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY", "META_BASE_URL"),
    base_url="https://api.meta.ai/v1", api_mode="codex_responses",
    # Images only on user turns: an image inside a role=tool message 400s.
    supports_vision=True, supports_vision_tool_messages=False,
    default_aux_model="muse-spark-1.2-contributor",
    # Muse spends completion budget on hidden reasoning first; low caps finish empty.
    default_max_tokens=16384,
    fallback_models=("muse-spark-1.2",),
))
