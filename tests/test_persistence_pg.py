"""Postgres-only behaviour. Needs a live server:

    GG_TEST_DATABASE_URL=postgresql://gg:gg@localhost:5432/gg_agent_test uv run pytest -m pg

Each test gets a throwaway schema, dropped afterwards.
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PG_DSN = os.getenv("GG_TEST_DATABASE_URL", "").strip()
pytestmark = [pytest.mark.pg, pytest.mark.skipif(not PG_DSN, reason="GG_TEST_DATABASE_URL not set")]


@pytest.fixture
def schema():
    import psycopg
    name = f"gg_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {name}")
    try:
        yield name
    finally:
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA {name} CASCADE")


def run(schema, body):
    from gg_agent.persistence.postgres import PostgresSessionStore

    async def main():
        store = PostgresSessionStore(PG_DSN, schema=schema)
        await store.open()
        try:
            return await body(store)
        finally:
            await store.close()
    return asyncio.run(main())


async def session_with(store, messages, sid=None):
    sid = sid or uuid.uuid4().hex[:12]
    owner = await store.get_user_by_email("pg@test.dev") or await store.create_user(uuid.uuid4().hex, "pg@test.dev", "h")
    await store.create_session(sid, owner_id=owner.id, source="cli", provider="p", model="m",
                               system_prompt=None, cwd="/w")
    await store.append_messages(sid, messages)
    return sid


def test_migrations_are_idempotent(schema):
    from gg_agent.persistence.migrate import discover_migrations, migrate

    async def body(store):
        again = await migrate(store.pool)
        async with store.pool.connection() as conn:
            rows = await (await conn.execute("SELECT version FROM schema_version ORDER BY version")).fetchall()
        return again, [r["version"] for r in rows]
    version, applied = run(schema, body)
    latest = discover_migrations()[-1][0]
    assert version == latest and applied == [v for v, _, _ in discover_migrations()]


def test_websearch_syntax(schema):
    async def body(store):
        a = await session_with(store, [{"role": "user", "content": "the blue whale migrates south"}])
        b = await session_with(store, [{"role": "user", "content": "a whale of a blue time"}])
        c = await session_with(store, [{"role": "user", "content": "red panda facts"}])
        ids = lambda hits: {h.session_id for h in hits}   # noqa: E731
        return (a, b, c,
                ids(await store.search('"blue whale"')),
                ids(await store.search("panda OR migrates")),
                ids(await store.search("whale -time")),
                ids(await store.search("whale blue")),
                await store.search('"unbalanced quote OR -'))
    a, b, c, phrase, either, excluded, both, garbage = run(schema, body)
    assert phrase == {a}
    assert either == {a, c}
    assert excluded == {a}
    assert both == {a, b}
    assert isinstance(garbage, list)           # malformed input never raises


def test_ranked_hits_have_highlighted_snippets(schema):
    async def body(store):
        await session_with(store, [{"role": "assistant", "content": "configure the psycopg pool with min_size"}])
        return await store.search("psycopg")
    hits = run(schema, body)
    assert len(hits) == 1 and "**psycopg**" in hits[0].snippet and hits[0].rank > 0


def test_ilike_fallback_escapes_wildcards(schema):
    async def body(store):
        sid = await session_with(store, [
            {"role": "user", "content": "the discount is 100% off"},
            {"role": "user", "content": "the discount is 1000 off"},
            {"role": "user", "content": "path gg_agent/persistence/postgres.py"},
        ])
        # "0% off" has no full-text match (the token is "100"), so this is the ILIKE
        # path — and an unescaped % would also match "1000 off".
        return sid, await store.search("0% off"), await store.search("persistence/postg")
    sid, percent, substring = run(schema, body)
    assert [h.snippet for h in percent] == ["the discount is 10**0% off**"]
    assert len(substring) == 1 and "**persistence/postg**" in substring[0].snippet


def test_huge_tool_output_inserts_and_round_trips(schema):
    big = "lorem ipsum " * 180_000            # ~2MB: over to_tsvector's 1MB limit uncapped
    async def body(store):
        sid = await session_with(store, [
            {"role": "user", "content": "dump it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "name": "read_file", "content": big},
        ])
        return await store.load_history(sid)
    history = run(schema, body)
    assert history[-1]["content"] == big


def test_concurrent_appends_to_one_session_stay_ordered(schema):
    async def body(store):
        sid = await session_with(store, [])
        async def writer(w):
            for i in range(10):
                await store.append_messages(sid, [{"role": "user", "content": f"{w}-{i}-a"},
                                                  {"role": "assistant", "content": f"{w}-{i}-b"}])
        await asyncio.gather(*(writer(w) for w in range(4)))
        return await store.get_messages(sid), await store.get_session(sid)
    rows, info = run(schema, body)
    assert len(rows) == 80 and info.message_count == 80
    # Each batch is contiguous: a lock-free interleaving would split an a/b pair.
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        assert first.content.endswith("-a") and second.content == first.content[:-1] + "b"
    assert [r.id for r in rows] == sorted(r.id for r in rows)


def test_unreachable_server_fails_fast():
    import psycopg

    from gg_agent.persistence.postgres import PostgresSessionStore

    async def main():
        store = PostgresSessionStore("postgresql://nobody:x@127.0.0.1:1/none")
        with pytest.raises(psycopg.OperationalError):
            await store.open()
    asyncio.run(asyncio.wait_for(main(), timeout=15))


def test_missing_schema_is_created_and_nothing_lands_in_public():
    """search_path skips a schema that doesn't exist; the store must create it
    first or the tables would silently go to public — next to other apps' tables."""
    import psycopg

    from gg_agent.persistence.postgres import PostgresSessionStore
    name = f"gg_fresh_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        before = conn.execute("SELECT to_regclass('public.users') IS NOT NULL").fetchone()[0]

    async def main():
        store = PostgresSessionStore(PG_DSN, schema=name)
        await store.open()
        try:
            await session_with(store, [{"role": "user", "content": "hi"}])
        finally:
            await store.close()
    try:
        asyncio.run(main())
        with psycopg.connect(PG_DSN, autocommit=True, client_encoding="utf8") as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (name,))}
            after = conn.execute("SELECT to_regclass('public.users') IS NOT NULL").fetchone()[0]
        assert {"users", "sessions", "messages", "schema_version"} <= tables
        assert after == before
    finally:
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")


def test_owner_is_enforced_by_the_database(schema):
    async def body(store):
        async with store.pool.connection() as conn:
            cols = await (await conn.execute(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'sessions' AND column_name = 'owner_id'"
            )).fetchone()
        with pytest.raises(KeyError):
            await store.create_session("s1", owner_id="ghost", source="cli", provider=None, model=None,
                                       system_prompt=None, cwd=None)
        return cols["is_nullable"]
    assert run(schema, body) == "NO"
