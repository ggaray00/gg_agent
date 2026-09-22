"""GitHub Copilot provider profile — an OAuth token exchanged for a short-lived API token.

Copilot needs a live credential *and* an account-specific base URL resolved per
turn (see ``providers/copilot_auth.py``); none of that leaks into the Agent, the
loop or the transport. It speaks the OpenAI Chat Completions shape, so it reuses
that transport unchanged.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ...reasoning_effort import clamp_effort, requested_effort
from .. import copilot_auth as ca
from .. import register_provider
from ..base import ProviderProfile

_COPILOT_EFFORTS = ("low", "medium", "high")


def _token_kind(token: str) -> str:
    """Token TYPE for display — never any of the secret itself."""
    for prefix in ("ghu_", "gho_", "ghp_", "github_pat_", "ghs_"):
        if token.startswith(prefix):
            return f"{prefix}…"
    return "opaque token"


class CopilotProfile(ProviderProfile):
    def resolve_api_key(self) -> str:
        try:
            return ca.get_copilot_credentials()[0]
        except Exception:
            return ""

    def has_credentials(self) -> bool:
        """Auto-detect on a COPILOT-SPECIFIC signal only, and never over the network.

        ``GH_TOKEN`` / ``GITHUB_TOKEN`` are usually exported for `gh` or CI, not for
        Copilot — auto-selecting this provider off one of them would hijack every run
        on a machine that has a perfectly good OPENAI_API_KEY. Both still work when
        Copilot is asked for explicitly (``-p copilot``).
        """
        if os.getenv("COPILOT_GITHUB_TOKEN", "").strip():
            return True
        return any(Path(os.path.expanduser(f)).is_file() for f in ca._CRED_FILES)

    def resolve_credentials(self) -> tuple[str, str]:
        # Re-read every turn: the exchanged token expires in ~30 minutes, and
        # enterprise accounts are not on the public base URL.
        api_token, base_url = ca.get_copilot_credentials()
        return api_token, base_url or self.base_url

    def credential_status(self) -> str:
        token, source = ca.resolve_github_token()
        if not token:
            return "no GitHub token found (run `gg-agent login`)"
        try:
            _, expires_at, base_url = ca.exchange_copilot_token(token)
        except Exception as exc:
            return f"GitHub token from {source} ({_token_kind(token)}) — exchange failed: {exc}"
        return (f"GitHub token from {source} ({_token_kind(token)}) → Copilot token valid for "
                f"{max(int(expires_at - time.time()), 0)}s @ {base_url}")

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                supports_reasoning: bool = False, **ctx: Any):
        # Only on an explicit effort: models that don't reason 400 on the field.
        effort = requested_effort(reasoning_config)
        if not supports_reasoning or not effort or effort == "none":
            return {}, {}
        clamped = clamp_effort(effort, _COPILOT_EFFORTS)
        return {"reasoning": {"effort": clamped if clamped in _COPILOT_EFFORTS else "medium"}}, {}


register_provider(CopilotProfile(
    name="copilot", aliases=("github-copilot", "github", "gh-copilot", "github-models", "github-model"),
    display_name="GitHub Copilot", auth_type="copilot",
    api_mode="chat_completions",
    env_vars=ca.COPILOT_ENV_VARS,
    base_url=ca.COPILOT_DEFAULT_BASE_URL,
    default_model="gpt-4.1",
    fallback_models=("gpt-4.1", "gpt-5", "claude-sonnet-4.5", "o4-mini"),
    default_headers=ca.copilot_request_headers(),
    default_aux_model="gpt-4.1-mini",
))
