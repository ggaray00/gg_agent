"""OpenAI Codex provider profile: the ChatGPT-subscription backend behind the Codex CLI.

No API key: it reuses the OAuth tokens the Codex CLI stores in
``$CODEX_HOME/auth.json`` (default ``~/.codex``) — sign in once with ``codex login``.
The access token is a JWT; it is refreshed with the stored refresh token shortly
before it expires, and the new pair is written back so the Codex CLI keeps working.

Mirrors hermes-agent: hermes_cli/auth_codex.py, agent/codex_headers.py
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from ...reasoning_effort import codex_supported_efforts
from .. import register_provider
from ..base import USER_AGENT, ProviderProfile

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
_REFRESH_MARGIN_SECONDS = 300
_lock = threading.Lock()


class CodexAuthError(RuntimeError):
    pass


def codex_auth_path() -> Path:
    return Path(os.getenv("CODEX_HOME", "").strip() or "~/.codex").expanduser() / "auth.json"


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:
        return {}


def _expiring(token: str) -> bool:
    exp = _jwt_claims(token).get("exp")
    return isinstance(exp, (int, float)) and exp - time.time() < _REFRESH_MARGIN_SECONDS


def _refresh(refresh_token: str) -> dict[str, Any]:
    body = json.dumps({"client_id": CODEX_OAUTH_CLIENT_ID, "grant_type": "refresh_token",
                       "refresh_token": refresh_token, "scope": "openid profile email"}).encode()
    req = urllib.request.Request(CODEX_OAUTH_TOKEN_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def get_codex_access_token() -> str:
    """A live access token from the Codex CLI's store, refreshing it when due."""
    path = codex_auth_path()
    with _lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CodexAuthError(f"{path} not found — sign in with `codex login`") from exc
        tokens = data.get("tokens") or {}
        access = str(tokens.get("access_token") or "")
        if not access:
            raise CodexAuthError(f"{path} has no ChatGPT tokens (API-key mode?) — run `codex login`")
        if _expiring(access) and tokens.get("refresh_token"):
            fresh = _refresh(str(tokens["refresh_token"]))
            tokens.update({k: fresh[k] for k in ("access_token", "refresh_token", "id_token") if fresh.get(k)})
            data["tokens"] = tokens
            data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(path)
            access = str(tokens["access_token"])
        return access


def codex_headers(access_token: str) -> dict[str, str]:
    """Identity + account headers the backend requires."""
    headers = {"User-Agent": USER_AGENT, "originator": "codex_cli_rs"}
    account = _jwt_claims(access_token).get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
    if isinstance(account, str) and account:
        headers["ChatGPT-Account-ID"] = account
    return headers


class OpenAICodexProfile(ProviderProfile):
    def resolve_credentials(self) -> tuple[str, str]:
        try:
            return get_codex_access_token(), self.base_url
        except Exception as exc:
            logger.debug("codex credentials unavailable: %s", exc)
            return "", self.base_url

    def has_credentials(self) -> bool:
        return codex_auth_path().is_file()

    def credential_status(self) -> str:
        try:
            token = get_codex_access_token()
        except Exception as exc:
            return str(exc)
        exp = _jwt_claims(token).get("exp")
        left = f", valid for {max(int(exp - time.time()), 0)}s" if isinstance(exp, (int, float)) else ""
        return f"ChatGPT OAuth from {codex_auth_path()}{left}"

    def client_headers(self, api_key: str) -> dict[str, str]:
        return codex_headers(api_key)

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        return codex_supported_efforts(model)

    def build_responses_extras(self, *, session_id: str | None = None, **ctx: Any) -> dict[str, Any]:
        # Keys starting with "_" are transport controls, not wire fields: the backend
        # only answers streamed requests and rejects max_output_tokens.
        extras: dict[str, Any] = {"_stream_only": True, "_omit_max_output_tokens": True}
        if session_id:
            extras["extra_headers"] = {"session_id": session_id}
            extras["prompt_cache_key"] = session_id
        return extras

    def fetch_models(self, **kwargs: Any):
        return None


register_provider(OpenAICodexProfile(
    name="openai-codex", aliases=("codex", "openai_codex", "chatgpt"), display_name="OpenAI Codex (ChatGPT)",
    description="GPT via a ChatGPT subscription — reuses `codex login`",
    api_mode="codex_responses", env_vars=(), base_url=CODEX_BASE_URL, auth_type="oauth_external",
    default_model="gpt-5.5", fallback_models=("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"),
    default_aux_model="gpt-5.4-mini",
))
