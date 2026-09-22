"""Ollama Cloud provider profile.

Top-level ``reasoning_effort`` accepts none|low|medium|high|max (``max`` is
undocumented but real); ``xhigh`` maps to ``max``.
"""

from typing import Any

from ...reasoning_effort import OLLAMA_CLOUD_EFFORTS, OLLAMA_CLOUD_OVERRIDES, clamp_effort
from .. import register_provider
from ..base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                supports_reasoning: bool = False, **ctx: Any):
        if not supports_reasoning or not isinstance(reasoning_config, dict):
            return {}, {}
        # Thinking defaults ON and extra_body.thinking is ignored: reasoning_effort
        # "none" is the only off switch.
        effort = (reasoning_config.get("effort") or "").strip().lower()
        if reasoning_config.get("enabled", True) is False or effort == "none":
            return {}, {"reasoning_effort": "none"}
        if not effort:
            return {}, {}
        clamped = clamp_effort(effort, OLLAMA_CLOUD_EFFORTS, OLLAMA_CLOUD_OVERRIDES)
        return {}, ({"reasoning_effort": clamped} if clamped in OLLAMA_CLOUD_EFFORTS else {})


register_provider(OllamaCloudProfile(
    name="ollama-cloud", aliases=("ollama_cloud",), display_name="Ollama Cloud",
    env_vars=("OLLAMA_API_KEY",), base_url="https://ollama.com/v1",
    default_aux_model="nemotron-3-nano:30b",
))
