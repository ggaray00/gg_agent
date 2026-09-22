"""xAI (Grok) provider profile — Responses API."""

import re
from typing import Any

from ...reasoning_effort import XAI_GROK46_EFFORTS, XAI_LEGACY_EFFORTS
from .. import register_provider
from ..base import USER_AGENT, ProviderProfile

_GROK_VERSION_RE = re.compile(r"grok-(\d+)(?:[.-](\d+))?")


class XAIProfile(ProviderProfile):
    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        # Grok 4.6+ accepts xhigh; older Grok tops out at high.
        match = _GROK_VERSION_RE.search((model or "").lower())
        version = (int(match.group(1)), int(match.group(2) or 0)) if match else (0, 0)
        return XAI_GROK46_EFFORTS if version >= (4, 6) else XAI_LEGACY_EFFORTS

    def build_responses_extras(self, *, session_id: str | None = None, **ctx: Any) -> dict[str, Any]:
        # xAI pins its prompt cache per backend server via this header.
        return {"extra_headers": {"x-grok-conv-id": session_id}} if session_id else {}


register_provider(XAIProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), display_name="xAI (Grok)",
    signup_url="https://console.x.ai/", api_mode="codex_responses",
    env_vars=("XAI_API_KEY", "XAI_BASE_URL"), base_url="https://api.x.ai/v1",
    default_headers={"User-Agent": USER_AGENT},
    supports_vision=True,
))
