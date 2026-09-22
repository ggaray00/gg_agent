"""Provider registry — the single source of truth for "which LLM are we talking to".

Every other layer (client construction, model listing, transport selection) reads
from a ``ProviderProfile`` here instead of keeping parallel data.

Profiles live one vendor per module under ``providers/plugins/``; each module
calls ``register_provider()`` at import. User plugins in
``$GG_HOME/plugins/model-providers/*.py`` load afterwards, so one with the same
name replaces the built-in (last writer wins).

Mirrors hermes-agent: providers/__init__.py, plugins/model-providers/
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
from collections.abc import Iterator

from .base import OMIT_TEMPERATURE, ProviderProfile

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}

# Auto-detection order when several providers have credentials. The original
# providers come first, in the order they always had, so adding a provider never
# changes which one an existing environment picks; everything else follows by name.
_DETECT_FIRST = ("anthropic", "copilot", "deepseek", "groq", "openai", "openrouter")


def register_provider(profile: ProviderProfile) -> ProviderProfile:
    """Add (or replace) a profile. Aliases resolve to the canonical name."""
    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name
    return profile


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up by canonical name or alias; None when unknown."""
    if not name:
        return None
    key = name.strip().lower()
    return _REGISTRY.get(_ALIASES.get(key, key))


def list_providers() -> list[ProviderProfile]:
    return sorted(_REGISTRY.values(), key=lambda p: p.name)


def iter_configured() -> Iterator[ProviderProfile]:
    """Profiles whose credentials are actually present, in auto-detect order."""
    rank = {name: i for i, name in enumerate(_DETECT_FIRST)}
    for profile in sorted(_REGISTRY.values(), key=lambda p: (rank.get(p.name, len(rank)), p.name)):
        if profile.has_credentials():
            yield profile


# ── Discovery ────────────────────────────────────────────────────────────────

def _discover_builtin() -> None:
    from . import plugins
    for info in sorted(pkgutil.iter_modules(plugins.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{plugins.__name__}.{info.name}")
        except Exception:
            logger.warning("provider plugin %s failed to load", info.name, exc_info=True)


def _discover_user() -> None:
    try:
        from ..home import get_gg_home
        folder = get_gg_home() / "plugins" / "model-providers"
    except Exception:
        return
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(f"gg_user_provider_{path.stem}", path)
            if spec and spec.loader:
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
        except Exception:
            logger.warning("user provider plugin %s failed to load", path, exc_info=True)


_discover_builtin()
_discover_user()

from .copilot_auth import get_copilot_credentials  # noqa: E402,F401  (re-export for tests/callers)
from .plugins.copilot import CopilotProfile  # noqa: E402  (re-export; needs discovery first)

__all__ = [
    "OMIT_TEMPERATURE", "ProviderProfile", "CopilotProfile", "register_provider",
    "get_provider_profile", "list_providers", "iter_configured",
]
