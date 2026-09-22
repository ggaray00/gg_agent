"""Microsoft Azure AI Foundry provider profile: OpenAI-compatible, per-resource base URL."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="azure-foundry", aliases=("azure", "azure-ai-foundry", "azure-ai"), display_name="Azure Foundry",
    description="Microsoft Foundry — OpenAI-compatible endpoint (set AZURE_FOUNDRY_BASE_URL)",
    signup_url="https://ai.azure.com/",
    env_vars=("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL"),
    base_url="",                            # per-resource; comes from AZURE_FOUNDRY_BASE_URL
))
