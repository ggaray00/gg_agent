"""Kilo Code gateway provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="kilocode", aliases=("kilo-code", "kilo", "kilo-gateway"), display_name="Kilo Code",
    env_vars=("KILOCODE_API_KEY",), base_url="https://api.kilo.ai/api/gateway",
    default_aux_model="google/gemini-3.6-flash",
))
