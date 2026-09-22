"""Arcee AI provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="arcee", aliases=("arcee-ai", "arceeai"), display_name="Arcee AI",
    env_vars=("ARCEEAI_API_KEY",), base_url="https://api.arcee.ai/api/v1",
))
