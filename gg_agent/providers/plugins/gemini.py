"""Google Gemini (AI Studio) provider profile, on Google's OpenAI-compatible endpoint.

Reasoning is translated to ``extra_body.google.thinking_config`` (snake_case),
which is how that endpoint takes Gemini's native thinkingConfig.
"""

from typing import Any

from .. import register_provider
from ..base import ProviderProfile
from ._common import gemini_openai_compat_extra_body


class GeminiProfile(ProviderProfile):
    def build_extra_body(self, *, session_id: str | None = None, **ctx: Any) -> dict[str, Any]:
        return gemini_openai_compat_extra_body(ctx.get("model") or "", ctx.get("reasoning_config"))

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        # The compat catalog answers "models/gemini-..."; requests want the bare id.
        return None if models is None else [m.removeprefix("models/") for m in models]


register_provider(GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), display_name="Google Gemini",
    signup_url="https://aistudio.google.com/apikey",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    default_model="gemini-3.6-flash", default_aux_model="gemini-3.6-flash",
    supports_vision=True,
))
