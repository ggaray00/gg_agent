"""Groq provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="groq",
    display_name="Groq",
    description="Groq — LPU-accelerated open models",
    signup_url="https://console.groq.com/keys",
    env_vars=("GROQ_API_KEY",),
    base_url="https://api.groq.com/openai/v1",
    default_model="llama-3.3-70b-versatile",
))
