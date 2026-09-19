"""Apply ``migrations/NNN_*.sql`` in order, once each.

No Alembic: a ``schema_version`` table holds the numbers already applied, and a
transaction-scoped advisory lock stops two processes migrating at once.
"""

from __future__ import annotations

import logging
import re
from importlib import resources

logger = logging.getLogger(__name__)

_LOCK_KEY = "gg_agent.migrate"


def discover_migrations() -> list[tuple[int, str, str]]:
    """``(version, filename, sql)`` for every migration, lowest version first."""
    found = []
    for entry in (resources.files(__package__) / "migrations").iterdir():
        match = re.match(r"(\d+)_.*\.sql$", entry.name)
        if match:
            found.append((int(match.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    return sorted(found)


async def migrate(pool) -> int:
    """Bring the schema up to date. Returns the resulting version."""
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_LOCK_KEY,))
        await conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version int NOT NULL)")
        cur = await conn.execute("SELECT coalesce(max(version), 0) AS version FROM schema_version")
        current = (await cur.fetchone())["version"]
        for version, name, sql in discover_migrations():
            if version <= current:
                continue
            logger.info("applying migration %s", name)
            await conn.execute(sql)
            await conn.execute("INSERT INTO schema_version (version) VALUES (%s)", (version,))
            current = version
    return current
