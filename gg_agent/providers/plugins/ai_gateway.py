"""Vercel AI Gateway provider profile: attribution headers + reasoning passthrough."""

from typing import Any

from .. import register_provider
from ..base import ProviderProfile
from ._common import ATTRIBUTION_HEADERS


class VercelAIGatewayProfile(ProviderProfile):
    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                supports_reasoning: bool = True, **ctx: Any):
        if not supports_reasoning:
            return {}, {}
        reasoning = dict(reasoning_config) if reasoning_config is not None else {"enabled": True, "effort": "medium"}
        return {"reasoning": reasoning}, {}


register_provider(VercelAIGatewayProfile(
    name="ai-gateway", aliases=("vercel", "vercel-ai-gateway", "ai_gateway", "aigateway"),
    display_name="Vercel AI Gateway",
    env_vars=("AI_GATEWAY_API_KEY",), base_url="https://ai-gateway.vercel.sh/v1",
    default_headers={k: v for k, v in ATTRIBUTION_HEADERS.items() if k != "User-Agent"},
    default_aux_model="google/gemini-3-flash",
))
