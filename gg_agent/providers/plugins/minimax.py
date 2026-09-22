"""MiniMax provider profiles (international, China).

Default routes use the Anthropic Messages API (base URLs end in /anthropic) with
Bearer auth. MiniMax-M3 can opt into the OpenAI-compatible https://api.minimax.io/v1
route (MINIMAX_BASE_URL + api_mode switch), which needs its own reasoning controls.
"""

from typing import Any
from urllib.parse import urlparse

from .. import register_provider
from ..base import ProviderProfile


def _is_minimax_global_openai_base_url(base_url: str | None) -> bool:
    parsed = urlparse(str(base_url or "").strip())
    return (parsed.hostname or "").lower() == "api.minimax.io" and parsed.path.rstrip("/").lower() == "/v1"


class MiniMaxProfile(ProviderProfile):
    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None, model: str | None = None,
                                base_url: str | None = None, **ctx: Any):
        """M3 on api.minimax.io/v1 keeps thinking inline unless ``reasoning_split`` is sent."""
        is_m3 = str(model or "").strip().lower() in {"minimax-m3", "minimax/minimax-m3"}
        if not _is_minimax_global_openai_base_url(base_url) or not is_m3:
            return {}, {}
        extra_body: dict[str, Any] = {"reasoning_split": True}
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            extra_body["thinking"] = {"type": "disabled"}
        elif reasoning_config is not None:
            extra_body["thinking"] = {"type": "adaptive"}
        return extra_body, {}


register_provider(MiniMaxProfile(
    name="minimax", aliases=("mini-max",), display_name="MiniMax", api_mode="anthropic_messages",
    env_vars=("MINIMAX_API_KEY", "MINIMAX_BASE_URL"), base_url="https://api.minimax.io/anthropic",
    bearer_auth=True, default_model="MiniMax-M3", default_aux_model="MiniMax-M3",
))
register_provider(MiniMaxProfile(
    name="minimax-cn", aliases=("minimax-china", "minimax_cn"), display_name="MiniMax (China)",
    api_mode="anthropic_messages",
    env_vars=("MINIMAX_CN_API_KEY", "MINIMAX_CN_BASE_URL"), base_url="https://api.minimaxi.com/anthropic",
    bearer_auth=True, default_model="MiniMax-M3", default_aux_model="MiniMax-M3",
))
