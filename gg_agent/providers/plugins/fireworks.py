"""Fireworks AI provider profile. Models are addressed by full catalog id
(``accounts/fireworks/models/<slug>``)."""

from .. import register_provider
from ..base import ProviderProfile
from ._common import ATTRIBUTION_HEADERS

register_provider(ProviderProfile(
    name="fireworks", aliases=("fireworks-ai", "fw"), display_name="Fireworks AI",
    description="Fireworks AI — OpenAI-compatible direct model API",
    signup_url="https://app.fireworks.ai/settings/users/api-keys",
    env_vars=("FIREWORKS_API_KEY",), base_url="https://api.fireworks.ai/inference/v1",
    default_headers=dict(ATTRIBUTION_HEADERS),
    default_aux_model="accounts/fireworks/models/glm-5p2",
    fallback_models=(
        "accounts/fireworks/models/kimi-k2p6", "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/kimi-k2p7-code",
    ),
))
