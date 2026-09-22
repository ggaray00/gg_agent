"""Native Anthropic provider profile."""

import json
import logging
import urllib.request

from .. import register_provider
from ..base import ProviderProfile

logger = logging.getLogger(__name__)


def is_oauth_token(key: str) -> bool:
    """Anthropic OAuth / setup tokens (Claude Code subscriptions), not Console API keys."""
    if not key or key.startswith("sk-ant-api"):
        return False
    return key.startswith(("sk-ant-", "eyJ", "cc-"))


class AnthropicProfile(ProviderProfile):
    """x-api-key + anthropic-version, not Bearer — also for the catalog."""

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        if not api_key:
            return None
        req = urllib.request.Request((base_url or self.base_url).rstrip("/") + "/v1/models?limit=1000")
        auth = ("Authorization", f"Bearer {api_key}") if is_oauth_token(api_key) else ("x-api-key", api_key)
        for k, v in (auth, ("anthropic-version", "2023-06-01"), ("Accept", "application/json")):
            req.add_header(k, v)
        if is_oauth_token(api_key):
            req.add_header("anthropic-beta", "oauth-2025-04-20")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(anthropic): %s", exc)
            return None
        return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]


register_provider(AnthropicProfile(
    name="anthropic", aliases=("claude", "claude-oauth", "claude-code"), display_name="Anthropic",
    description="Anthropic — Claude via the Messages API", signup_url="https://console.anthropic.com/",
    api_mode="anthropic_messages",           # <- different transport, same loop
    # API key first; OAuth tokens (Claude subscriptions) authenticate with Bearer.
    env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL"),
    base_url="https://api.anthropic.com",
    default_model="claude-sonnet-5",
    fallback_models=("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"),
    default_max_tokens=8192,                 # Anthropic requires max_tokens
    default_aux_model="claude-haiku-4-5-20251001",
    supports_vision=True,
))
