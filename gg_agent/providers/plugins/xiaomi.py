"""Xiaomi MiMo provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), display_name="Xiaomi MiMo",
    env_vars=("XIAOMI_API_KEY",), base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,            # /v1/models 401s even with a valid key
    supports_vision=True,                   # mimo-v2-omni
    supports_vision_tool_messages=False,    # list-type tool content -> 400 "text is not set"
))
