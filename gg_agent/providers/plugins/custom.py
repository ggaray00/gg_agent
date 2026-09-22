"""Custom endpoint provider profile: any OpenAI-compatible server (vLLM,
llama.cpp, LM Studio, SGLang, a proxy...). Point it with CUSTOM_BASE_URL."""

from typing import Any
from urllib.parse import urlparse

from ...reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort
from .. import register_provider
from ..base import ProviderProfile


def looks_like_ollama_endpoint(base_url: str | None) -> bool:
    """Explicit Ollama signatures only (port 11434 or an ``ollama`` host label):
    ``think`` is Ollama-native and strict hosts 422 on it."""
    raw = (base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:
        if parsed.port == 11434:
            return True
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return bool(host) and (host == "ollama.com" or host.endswith(".ollama.com") or "ollama" in host.split("."))


class CustomProfile(ProviderProfile):
    def has_credentials(self) -> bool:
        return bool(self.resolve_base_url())

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                ollama_num_ctx: int | None = None, **ctx: Any):
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if ollama_num_ctx:
            extra_body["options"] = {"num_ctx": ollama_num_ctx}
        # disabled -> reasoning_effort="none" (Ollama's /v1 ignores extra_body.think)
        # plus think=False on Ollama URLs; an effort -> clamped to the OpenAI-compat
        # wire (vLLM/SGLang top out at "max"); enabled without effort -> server default.
        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort == "none" or reasoning_config.get("enabled", True) is False:
                top_level["reasoning_effort"] = "none"
                if looks_like_ollama_endpoint(ctx.get("base_url")):
                    extra_body["think"] = False
            elif effort:
                top_level["reasoning_effort"] = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)
        return extra_body, top_level

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        if not (base_url or self.resolve_base_url()):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url or self.resolve_base_url(), timeout=timeout)


register_provider(CustomProfile(
    name="custom", aliases=("local", "vllm", "llamacpp", "llama.cpp", "llama-cpp", "lmstudio"),
    display_name="Custom endpoint", description="Any OpenAI-compatible server (set CUSTOM_BASE_URL)",
    env_vars=("CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
    base_url="",
    # A floor only (the caller's max_tokens wins): without one some servers
    # fall back to tiny internal defaults and truncate after a few tokens.
    default_max_tokens=65536,
))
