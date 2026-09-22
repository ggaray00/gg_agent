"""OpenAI provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="openai",
    display_name="OpenAI",
    description="OpenAI — GPT models via Chat Completions",
    signup_url="https://platform.openai.com/api-keys",
    env_vars=("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    base_url="https://api.openai.com/v1",
    default_model="gpt-4.1",
    fallback_models=("gpt-4.1", "gpt-4.1-mini", "o4-mini"),
    default_aux_model="gpt-4.1-mini",
    supports_vision=True,
    supports_prompt_cache_key=True,
))
