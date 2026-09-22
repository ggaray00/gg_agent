"""Google Vertex AI provider profile: Gemini via Google Cloud's OpenAI-compatible endpoint.

Auth is OAuth2 (service-account JSON or Application Default Credentials), not a
static key: ``resolve_credentials`` mints a short-lived access token every turn,
which the Agent swaps in when it changes. Needs ``pip install google-auth``.

Settings: VERTEX_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS (SA JSON;
the former wins; neither = ADC), VERTEX_PROJECT_ID (else the credentials'
project), VERTEX_REGION (default "global").

Mirrors hermes-agent: agent/vertex_adapter.py
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from .. import register_provider
from ..base import ProviderProfile
from ._common import gemini_openai_compat_extra_body

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_lock = threading.Lock()
_cache: dict[str, tuple[Any, str | None]] = {}      # credentials path ("" = ADC) -> (creds, project)


def _credentials_path() -> str:
    for var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = os.getenv(var, "").strip()
        if path and os.path.exists(path):
            return path
    return ""


def vertex_base_url(project: str, region: str) -> str:
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{region}/endpoints/openapi"


def get_vertex_credentials() -> tuple[str, str]:
    """``(access_token, base_url)``; raises when google-auth or credentials are missing."""
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account

    path = _credentials_path()
    with _lock:
        cached = _cache.get(path)
        if cached is None:
            if path:
                creds = service_account.Credentials.from_service_account_file(path, scopes=_SCOPES)
                cached = (creds, creds.project_id)
            else:
                cached = google.auth.default(scopes=_SCOPES)
            _cache[path] = cached
        creds, project = cached
        expiry = getattr(creds, "expiry", None)
        if (not getattr(creds, "token", None) or getattr(creds, "expired", False)
                or (expiry is not None and expiry.timestamp() - time.time() < 300)):
            creds.refresh(google.auth.transport.requests.Request())
    project = os.getenv("VERTEX_PROJECT_ID", "").strip() or project
    if not project:
        raise RuntimeError("Vertex AI: no project id (set VERTEX_PROJECT_ID)")
    region = os.getenv("VERTEX_REGION", "").strip() or "global"
    return creds.token, vertex_base_url(project, region)


class VertexProfile(ProviderProfile):
    def resolve_credentials(self) -> tuple[str, str]:
        try:
            return get_vertex_credentials()
        except Exception as exc:
            logger.debug("vertex credentials unavailable: %s", exc)
            return "", ""

    def has_credentials(self) -> bool:
        return bool(_credentials_path() or os.getenv("VERTEX_PROJECT_ID", "").strip())

    def credential_status(self) -> str:
        try:
            _, base_url = get_vertex_credentials()
        except ImportError:
            return "google-auth not installed (pip install google-auth)"
        except Exception as exc:
            return f"no usable credentials: {exc}"
        return f"OAuth2 via {_credentials_path() or 'application default credentials'} @ {base_url}"

    def build_extra_body(self, *, session_id: str | None = None, **ctx: Any) -> dict[str, Any]:
        return gemini_openai_compat_extra_body(ctx.get("model") or "", ctx.get("reasoning_config"))

    def fetch_models(self, **kwargs: Any):
        return None                          # no /models route on the compat endpoint


register_provider(VertexProfile(
    name="vertex", aliases=("google-vertex", "vertex-ai", "gcp-vertex"), display_name="Google Vertex AI",
    env_vars=(), auth_type="vertex",
    base_url="https://aiplatform.googleapis.com",   # the real one is computed per project/region
    default_model="google/gemini-3.6-flash", default_aux_model="google/gemini-3.6-flash",
    supports_vision=True,
))
