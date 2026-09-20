"""The gg home directory, and the working directory.

Two different paths, kept apart the way hermes-agent keeps them apart:

* the *home* (``~/.gg_agent``) holds config and state — it is per user, not per
  project, and gg creates it;
* the *working directory* is wherever gg was launched from, and it is what the
  tools act on. It stays the user's project, never the home.

Mirrors hermes-agent: hermes_constants.get_hermes_home() /
hermes_cli.config.ensure_hermes_home() / tools.terminal_tool_config._safe_getcwd().
"""

from __future__ import annotations

import os
from pathlib import Path

# Created up front so a writer never has to think about whether its parent exists.
_SUBDIRS = ("sessions", "logs", "memories", "skills")

# Homes already built, so the mkdir/chmod syscalls happen once per process even
# though ensure_gg_home() sits on startup paths that may run repeatedly (tests,
# embedded use). Keyed by path string: a test that moves GG_HOME gets a fresh key.
_ensured: set[str] = set()


def get_gg_home() -> Path:
    """Home for config and state: ``$GG_HOME`` → ``~/.gg_agent``."""
    val = os.getenv("GG_HOME", "").strip()
    return Path(val).expanduser() if val else Path.home() / ".gg_agent"


def ensure_gg_home() -> Path:
    """Create the home skeleton, readable only by its owner. Idempotent."""
    home = get_gg_home()
    key = str(home)
    # is_dir() as well as the memo: the directory can be deleted underneath us.
    if key in _ensured and home.is_dir():
        return home
    home.mkdir(parents=True, exist_ok=True)
    # Sessions and logs hold conversation text; keep the whole tree at 0700.
    home.chmod(0o700)
    for name in _SUBDIRS:
        subdir = home / name
        subdir.mkdir(exist_ok=True)
        subdir.chmod(0o700)
    _ensured.add(key)
    return home


def reset_ensured_cache() -> None:
    """Forget which homes have been built (for tests that move ``$GG_HOME``)."""
    _ensured.clear()


def get_working_dir() -> str:
    """Where the tools run: the launch directory, as an absolute path.

    ``os.getcwd()`` raises if the directory has been deleted out from under the
    process, or — on macOS without Full Disk Access — if it is TCC-protected.
    ``$GG_CWD`` is the anchor main() recorded at startup; ``~`` is the last
    resort (the user's home, not ``~/.gg_agent``).
    """
    try:
        return os.getcwd()
    except (FileNotFoundError, PermissionError):
        return os.getenv("GG_CWD") or os.path.expanduser("~")
