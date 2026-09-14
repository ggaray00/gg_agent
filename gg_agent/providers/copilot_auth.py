"""GitHub Copilot authentication.

Copilot is a two-stage credential, which is why it needs its own module while
every other provider is one env var:

    1. a long-lived GitHub OAuth token (``ghu_*`` / ``gho_*`` / ``github_pat_*``)
       — from an env var, the VS Code / Copilot CLI on-disk store, or `gh auth token`;
    2. exchanged at ``api.github.com/copilot_internal/v2/token`` for a SHORT-LIVED
       (~30 min) Copilot API token, which is what actually goes in the
       ``Authorization: Bearer`` header of every chat request.

Stage 2 also tells you the account's real base URL — Copilot Enterprise and
proxied accounts are NOT on ``api.githubcopilot.com`` — so the exchange result
carries both halves of the credential.

Because the exchanged token expires mid-session, ``CopilotProfile.resolve_credentials()``
is re-read before every turn and the Agent rebuilds its client when it changes.

Mirrors hermes-agent: hermes_cli/copilot_auth.py, plugins/model-providers/copilot/
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# VS Code's GitHub App client ID. It mints ghu_* tokens, which the exchange accepts
# for every model the account can see. Other app IDs mint gho_* tokens that work for
# some accounts but 404 on enterprise-only models.
COPILOT_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
COPILOT_DEFAULT_BASE_URL = "https://api.githubcopilot.com"

# Classic PATs are rejected by the Copilot API — catch it here with a useful
# message rather than at request time with an opaque 401.
_CLASSIC_PAT_PREFIX = "ghp_"

# On-disk OAuth stores written by VS Code's Copilot extension and the Copilot CLI.
_CRED_FILES = (
    "~/.config/github-copilot/apps.json",
    "~/.config/github-copilot/hosts.json",
    "~/.copilot/config.json",
)

# Exchange endpoint + headers (matching VS Code / the Copilot CLI).
_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_EDITOR_VERSION = "vscode/1.104.1"
_EXCHANGE_USER_AGENT = "GitHubCopilotChat/0.26.7"

_JWT_REFRESH_MARGIN_SECONDS = 120          # refresh 2 min before expiry
_EXCHANGE_FAILURE_TTL_TRANSIENT = 60.0     # network blip: retry soon
_EXCHANGE_FAILURE_TTL_PERMANENT = 1800.0   # 401/403/404: won't heal on its own
_PERMANENT_HTTP_STATUSES = frozenset({401, 403, 404})

# token fingerprint -> (api_token, expires_at, base_url)
_jwt_cache: dict[str, tuple[str, float, str]] = {}
_failure_cache: dict[str, float] = {}
_lock = threading.Lock()


class CopilotAuthError(RuntimeError):
    """Raised when no usable Copilot credential can be produced."""


# ── Stage 1: find the GitHub OAuth token ─────────────────────────────────────

def validate_github_token(token: str) -> tuple[bool, str]:
    token = (token or "").strip()
    if not token:
        return False, "Empty token"
    if token.startswith(_CLASSIC_PAT_PREFIX):
        return False, (
            "Classic Personal Access Tokens (ghp_*) are not accepted by the Copilot API. "
            "Use `gg-agent login` (OAuth device flow), a fine-grained PAT (github_pat_*) "
            "with the Copilot Requests permission, or sign in via VS Code / the Copilot CLI."
        )
    return True, "OK"


def _token_from_env() -> tuple[str, str]:
    for var in COPILOT_ENV_VARS:
        value = os.getenv(var, "").strip()
        if not value:
            continue
        valid, msg = validate_github_token(value)
        if valid:
            return value, var
        logger.warning("Token in %s is not usable: %s", var, msg)
    return "", ""


def _oauth_token_from_blob(blob: object) -> str:
    """Pull an OAuth token out of a Copilot credential store.

    The stores are keyed by ``<host>:<app id>`` and hold ``{"user", "oauth_token"}``;
    the Copilot CLI variant nests tokens under ``copilotTokens``. Rather than pin one
    layout, walk the structure and take the first plausible token — the schema has
    changed more than once across Copilot releases.
    """
    if isinstance(blob, str):
        return blob.strip() if blob.strip().startswith(("ghu_", "gho_", "github_pat_")) else ""
    if isinstance(blob, dict):
        for field in ("oauth_token", "token", "access_token"):
            value = blob.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in blob.values():
            found = _oauth_token_from_blob(value)
            if found:
                return found
    return ""


def _token_from_disk() -> tuple[str, str]:
    for raw_path in _CRED_FILES:
        path = Path(os.path.expanduser(raw_path))
        try:
            if not path.is_file() or path.stat().st_size <= 2:
                continue
            # The Copilot CLI writes JSONC — strip whole-line // comments.
            text = "\n".join(line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                             if not line.lstrip().startswith("//"))
            token = _oauth_token_from_blob(json.loads(text) if text.strip() else {})
        except Exception as exc:
            logger.debug("could not read %s: %s", raw_path, exc)
            continue
        if token and validate_github_token(token)[0]:
            return token, raw_path
    return "", ""


def _token_from_gh_cli() -> tuple[str, str]:
    binary = shutil.which("gh") or next(
        (p for p in ("/opt/homebrew/bin/gh", "/usr/local/bin/gh") if os.access(p, os.X_OK)), None)
    if not binary:
        return "", ""
    # gh must not echo back the very env vars we already checked, nor prompt.
    env = {k: v for k, v in os.environ.items() if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}
    env.setdefault("GH_PROMPT_DISABLED", "1")
    try:
        proc = subprocess.run([binary, "auth", "token"], capture_output=True, text=True,
                              timeout=5, env=env, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("gh auth token failed: %s", exc)
        return "", ""
    token = proc.stdout.strip() if proc.returncode == 0 else ""
    return (token, "gh auth token") if token and validate_github_token(token)[0] else ("", "")


def resolve_github_token() -> tuple[str, str]:
    """``(token, source)`` for the GitHub OAuth token, or ``("", "")``.

    Order matches the Copilot CLI: explicit env var wins, then the on-disk store
    written by VS Code / the Copilot CLI, then `gh auth token`.
    """
    for finder in (_token_from_env, _token_from_disk, _token_from_gh_cli):
        token, source = finder()
        if token:
            return token, source
    return "", ""


# ── Stage 2: exchange for a Copilot API token ────────────────────────────────

def _fingerprint(raw_token: str) -> str:
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


def _fresh(entry: tuple[str, float, str] | None) -> bool:
    return bool(entry) and time.time() < entry[1] - _JWT_REFRESH_MARGIN_SECONDS


def _derive_base_url(api_token: str) -> str:
    """Copilot host from the token's ``proxy-ep=proxy.<host>`` field (→ ``api.<host>``).

    Enterprise and proxied accounts are not on the public host; sending their
    requests there fails every turn with an unhelpful error.
    """
    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", api_token or "")
    if not match:
        return ""
    host = re.sub(r"^https?://", "", match.group(1), count=1).rstrip("/")
    # Computed outside the f-string: a backslash inside an f-string expression is a
    # syntax error before Python 3.12, and this package supports 3.10.
    api_host = re.sub(r"^proxy\.", "api.", host, count=1)
    return f"https://{api_host}"


def exchange_copilot_token(raw_token: str, *, timeout: float = 10.0) -> tuple[str, float, str]:
    """Exchange a GitHub OAuth token for ``(api_token, expires_at, base_url)``.

    The result is cached in-process until close to expiry. A failure is negatively
    cached too: without that, a permanently-rejected token re-runs the network call
    on every single turn.
    """
    fingerprint = _fingerprint(raw_token)
    cached = _jwt_cache.get(fingerprint)
    if _fresh(cached):
        return cached

    with _lock:
        cached = _jwt_cache.get(fingerprint)          # another thread may have just done it
        if _fresh(cached):
            return cached
        fail_until = _failure_cache.get(fingerprint, 0.0)
        if time.time() < fail_until:
            raise CopilotAuthError(
                f"Copilot token exchange failed recently; not retrying for "
                f"{int(fail_until - time.time())}s")

        request = urllib.request.Request(
            _TOKEN_EXCHANGE_URL, method="GET",
            headers={"Authorization": f"token {raw_token}",
                     "User-Agent": _EXCHANGE_USER_AGENT,
                     "Accept": "application/json",
                     "Editor-Version": _EDITOR_VERSION})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            ttl = (_EXCHANGE_FAILURE_TTL_PERMANENT if exc.code in _PERMANENT_HTTP_STATUSES
                   else _EXCHANGE_FAILURE_TTL_TRANSIENT)
            _failure_cache[fingerprint] = time.time() + ttl
            hint = (" — the token may lack Copilot access, or your Copilot subscription "
                    "is inactive. Try `gg-agent login`.") if exc.code in _PERMANENT_HTTP_STATUSES else ""
            raise CopilotAuthError(f"Copilot token exchange returned HTTP {exc.code}{hint}") from exc
        except Exception as exc:
            _failure_cache[fingerprint] = time.time() + _EXCHANGE_FAILURE_TTL_TRANSIENT
            raise CopilotAuthError(f"Copilot token exchange failed: {exc}") from exc

        api_token = str(data.get("token") or "")
        if not api_token:
            _failure_cache[fingerprint] = time.time() + _EXCHANGE_FAILURE_TTL_TRANSIENT
            raise CopilotAuthError("Copilot token exchange returned an empty token")

        expires_at = float(data.get("expires_at") or 0) or (time.time() + 1800)
        endpoints = data.get("endpoints")
        base_url = (str(endpoints.get("api") or "").strip().rstrip("/")
                    if isinstance(endpoints, dict) else "")
        base_url = base_url or _derive_base_url(api_token) or COPILOT_DEFAULT_BASE_URL

        _failure_cache.pop(fingerprint, None)
        entry = (api_token, expires_at, base_url)
        _jwt_cache[fingerprint] = entry
        logger.debug("Copilot token exchanged; expires in %ds, base_url=%s",
                     int(expires_at - time.time()), base_url)
        return entry


def get_copilot_credentials() -> tuple[str, str]:
    """``(api_token, base_url)`` ready for the chat endpoint.

    Raises ``CopilotAuthError`` when no GitHub token can be found at all; a token
    that merely fails to exchange falls back to being sent raw, which works for
    some individual accounts.
    """
    raw_token, source = resolve_github_token()
    if not raw_token:
        raise CopilotAuthError(
            "No GitHub Copilot credential found. Either:\n"
            "  • run `gg-agent login` (OAuth device flow), or\n"
            "  • sign in with VS Code / the Copilot CLI, or\n"
            "  • export COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN"
        )
    try:
        api_token, _, base_url = exchange_copilot_token(raw_token)
        return api_token, base_url
    except CopilotAuthError as exc:
        logger.debug("exchange failed (source=%s), falling back to the raw token: %s", source, exc)
        return raw_token, COPILOT_DEFAULT_BASE_URL


def copilot_request_headers(*, is_agent_turn: bool = True) -> dict[str, str]:
    """Headers the Copilot API requires on every chat request.

    Without ``Copilot-Integration-Id`` the API rejects the request outright; the
    rest is editor attribution it expects to see.
    """
    return {
        "Editor-Version": _EDITOR_VERSION,
        "Editor-Plugin-Version": "gg-agent/0.1.0",
        "Copilot-Integration-Id": "vscode-chat",
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
        "User-Agent": "GitHubCopilotChat/0.26.7",
    }


# ── Interactive login (OAuth device code flow) ───────────────────────────────

def _post_form(url: str, fields: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=urllib.parse.urlencode(fields).encode(),
        headers={"Accept": "application/json", "User-Agent": "gg-agent/0.1.0",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def device_code_login(*, host: str = "github.com", timeout_seconds: float = 300) -> str | None:
    """Run GitHub's OAuth device-code flow and return the ``ghu_*`` token.

    The token is printed for the caller to export — gg-agent deliberately does not
    write to the credential stores that VS Code and the Copilot CLI own.
    """
    domain = host.rstrip("/")
    try:
        device = _post_form(f"https://{domain}/login/device/code",
                            {"client_id": COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"}, 15)
    except Exception as exc:
        print(f"  ✗ Could not start device authorization: {exc}")
        return None

    user_code = device.get("user_code", "")
    device_code = device.get("device_code", "")
    verification_uri = device.get("verification_uri", f"https://{domain}/login/device")
    interval = max(int(device.get("interval", 5)), 1)
    if not (user_code and device_code):
        print("  ✗ GitHub did not return a device code.")
        return None

    print(f"\n  Open: {verification_uri}\n  Enter code: {user_code}\n")
    print("  Waiting for authorization", end="", flush=True)

    poll = {"client_id": COPILOT_OAUTH_CLIENT_ID, "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(interval + 3)
        try:
            result = _post_form(f"https://{domain}/login/oauth/access_token", poll, 10)
        except Exception:
            print(".", end="", flush=True)
            continue
        if result.get("access_token"):
            print(" ✓")
            return result["access_token"]
        error = result.get("error", "")
        if error == "slow_down":
            server_interval = result.get("interval")
            interval = int(server_interval) if isinstance(server_interval, (int, float)) and server_interval > 0 else interval + 5
        if error in ("authorization_pending", "slow_down"):
            print(".", end="", flush=True)
            continue
        if error:
            print(f"\n  ✗ Authorization failed: {error}")
            return None
    print("\n  ✗ Timed out waiting for authorization.")
    return None
