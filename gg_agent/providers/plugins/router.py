"""Ramp Router (router.com) provider profile: a Responses-only LLM gateway.

Router 400s on ``reasoning.effort`` levels outside a model's published vocabulary
and on any reasoning field for non-reasoning models. The per-model efforts map
comes from ``GET /v1/models``; it is cached in memory and mirrored to disk, and
the request path only ever reads the cache (a cold cache means "unknown").
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ...reasoning_effort import EFFORT_LADDER
from .. import register_provider
from ..base import USER_AGENT, ProviderProfile

logger = logging.getLogger(__name__)

ROUTER_DEFAULT_BASE_URL = "https://api.router.com/v1"
_DISK_TTL_SECONDS = 24 * 60 * 60

#: model id -> accepted effort levels. [] = no reasoning fields; absent = unknown.
_efforts_cache: dict[str, list[str]] | None = None
_efforts_lock = threading.Lock()
_disk_checked = False


def _dig(obj: Any, *keys: str) -> Any:
    for key in keys:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def parse_efforts(items: Any) -> dict[str, list[str]] | None:
    """A ``/v1/models`` ``data`` array -> the efforts map (None if unusable).
    Levels the ladder doesn't know are dropped: clamping can't place them."""
    if not isinstance(items, list):
        return None
    efforts_by_id: dict[str, list[str]] = {}
    for item in items:
        mid = str(item.get("id") or "").strip() if isinstance(item, dict) else ""
        reasoning = _dig(item, "router", "capabilities", "reasoning")
        if not mid or not isinstance(reasoning, dict):
            continue
        if reasoning.get("supported") is False:
            efforts_by_id[mid] = []
            continue
        levels = [str(e.get("value") or "").strip() for e in reasoning.get("efforts") or [] if isinstance(e, dict)]
        levels = [lv for lv in levels if lv in EFFORT_LADDER]
        if levels:
            efforts_by_id[mid] = levels
    return efforts_by_id or None


def _disk_path() -> Path | None:
    try:
        from ...home import get_gg_home
        return get_gg_home() / "cache" / "router_catalog.json"
    except Exception:
        return None


def _seed(items: Any) -> None:
    global _efforts_cache
    parsed = parse_efforts(items)
    if parsed is None:
        return
    with _efforts_lock:
        _efforts_cache = parsed
    path = _disk_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")       # write-then-rename: readers never see a torn file
        tmp.write_text(json.dumps({"ts": time.time(), "efforts": parsed}), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("router: caps disk mirror write failed: %s", exc)


def _cache_only() -> dict[str, list[str]] | None:
    """Memory, else the disk mirror (read once per process). Never HTTP."""
    global _efforts_cache, _disk_checked
    with _efforts_lock:
        if _efforts_cache is not None or _disk_checked:
            return _efforts_cache
        _disk_checked = True
    path = _disk_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path else {}
    except Exception:
        return None
    efforts = data.get("efforts")
    # A stale verdict still beats none; it is refreshed on the next --list-models.
    if isinstance(efforts, dict) and efforts:
        with _efforts_lock:
            _efforts_cache = _efforts_cache or {str(k): list(v) for k, v in efforts.items() if isinstance(v, list)}
        if time.time() - float(data.get("ts") or 0) > _DISK_TTL_SECONDS:
            logger.debug("router: efforts cache is stale; run --list-models -p router to refresh")
    return _efforts_cache


class RouterProfile(ProviderProfile):
    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0):
        """Live, key-scoped catalog; the same payload seeds the efforts cache.
        Deduped, not sorted: Router's listing order is deliberate."""
        import urllib.request
        req = urllib.request.Request((base_url or self.resolve_base_url()).rstrip("/") + "/models")
        key = api_key or self.resolve_api_key()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", USER_AGENT)      # Router's WAF rejects the default urllib UA
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("router: catalog fetch failed: %s", exc)
            return None
        items = data if isinstance(data, list) else data.get("data", [])
        _seed(items)
        return list(dict.fromkeys(str(i["id"]) for i in items if isinstance(i, dict) and i.get("id"))) or None

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        efforts = _cache_only()
        mid = str(model or "").strip()
        return tuple(efforts[mid]) if efforts and mid in efforts else None


register_provider(RouterProfile(
    name="router", aliases=("ramp-router", "ramp", "router.com"), display_name="Ramp Router",
    description="Ramp Router (router.com) — routes each request to the cheapest model that clears your bar",
    signup_url="https://app.router.com/keys", api_mode="codex_responses",
    env_vars=("RAMP_ROUTER_API_KEY", "ROUTER_API_KEY", "RAMP_ROUTER_BASE_URL"),
    base_url=ROUTER_DEFAULT_BASE_URL,
    # Router attributes clients by UA prefix; its WAF rejects default UAs.
    default_headers={"User-Agent": USER_AGENT},
    supports_vision=True, default_aux_model="gpt-5.4-mini",
    fallback_models=(),                      # account-scoped ids: use --list-models
))
