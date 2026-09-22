"""NovitaAI provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="novita", aliases=("novita-ai", "novitaai"), display_name="NovitaAI",
    description="NovitaAI — AI-native cloud for builders and agents",
    signup_url="https://novita.ai/settings/key-management",
    env_vars=("NOVITA_API_KEY", "NOVITA_BASE_URL"), base_url="https://api.novita.ai/openai/v1",
    default_aux_model="deepseek/deepseek-v3-0324",
    fallback_models=(
        "moonshotai/kimi-k2.5", "minimax/minimax-m2.7", "zai-org/glm-5", "deepseek/deepseek-v3-0324",
        "deepseek/deepseek-r1-0528", "qwen/qwen3-235b-a22b-fp8",
    ),
))
