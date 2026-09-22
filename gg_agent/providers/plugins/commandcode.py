"""CommandCode provider profiles: ``commandcode`` (Chat Completions) and
``commandcode-anthropic`` (Anthropic Messages, Bearer auth). Same key, same base URL."""

from .. import register_provider
from ..base import ProviderProfile

_BASE = "https://api.commandcode.ai/provider/v1"


class CommandCodeProfile(ProviderProfile):
    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        # The catalog is public: never send the key to it.
        return super().fetch_models(api_key=None, base_url=base_url, timeout=timeout)


class CommandCodeAnthropicProfile(CommandCodeProfile):
    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        return None if models is None else [m for m in models if m.startswith("claude-")]


register_provider(CommandCodeProfile(
    name="commandcode", aliases=("commandcode-chat",), display_name="CommandCode",
    description="CommandCode — 20+ models via OpenAI-compatible API", signup_url="https://commandcode.ai/",
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_BASE_URL"), base_url=_BASE, models_url=f"{_BASE}/models",
    fallback_models=(
        "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "Qwen/Qwen3.7-Max", "Qwen/Qwen3.6-Plus",
        "moonshotai/Kimi-K2.6", "zai-org/GLM-5.1", "MiniMaxAI/MiniMax-M2.7", "stepfun/Step-3.5-Flash",
        "xiaomi/mimo-v2.5-pro", "google/gemini-3.5-flash", "gpt-5.5",
    ),
    default_aux_model="deepseek/deepseek-v4-flash",
))
register_provider(CommandCodeAnthropicProfile(
    name="commandcode-anthropic", aliases=("commandcode-claude",), display_name="CommandCode (Anthropic)",
    description="CommandCode — Claude models via Anthropic Messages API", signup_url="https://commandcode.ai/",
    api_mode="anthropic_messages", bearer_auth=True,
    env_vars=("COMMANDCODE_API_KEY", "COMMANDCODE_ANTHROPIC_BASE_URL"), base_url=_BASE, models_url=f"{_BASE}/models",
    fallback_models=("claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"),
    default_aux_model="claude-haiku-4-5-20251001",
))
