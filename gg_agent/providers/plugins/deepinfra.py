"""DeepInfra provider profile (chat surface)."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="deepinfra", aliases=("deep-infra", "deepinfra-ai"), display_name="DeepInfra",
    description="DeepInfra — 100+ open models, pay-per-use", signup_url="https://deepinfra.com/dash/api_keys",
    env_vars=("DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL"), base_url="https://api.deepinfra.com/v1/openai",
    default_max_tokens=None,                # DeepInfra applies its per-model limit
    default_aux_model="deepseek-ai/DeepSeek-V4-Flash",
    # Empty on purpose: the live catalog is the source of truth (`--list-models`).
    fallback_models=(),
))
