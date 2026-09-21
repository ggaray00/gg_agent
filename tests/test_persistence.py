"""Session persistence: store contract (in-memory always, Postgres when configured),
serialization, and the loop/Agent integration. No network, no database needed."""
import asyncio
import concurrent.futures
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.aio import run_sync
from gg_agent.persistence import InMemorySessionStore, UserInfo, get_default_store, make_title
from gg_agent.persistence.serialize import message_to_row, repair_for_resume, row_to_message
from gg_agent.persistence.users import authenticate, hash_password, register_user, verify_password
from gg_agent.tools.registry import registry
from gg_agent.transports.base import ProviderTransport
from gg_agent.transports.types import NormalizedResponse, ToolCall, Usage

PG_DSN = os.getenv("GG_TEST_DATABASE_URL", "").strip()


# ── Store factories: the same contract runs against every implementation ────

@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.pg)])
def make_store(request):
    if request.param == "memory":
        yield InMemorySessionStore
        return
    if not PG_DSN:
        pytest.skip("GG_TEST_DATABASE_URL not set")
    import psycopg

    from gg_agent.persistence.postgres import PostgresSessionStore
    schema = f"gg_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    try:
        yield lambda: PostgresSessionStore(PG_DSN, schema=schema)
    finally:
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA {schema} CASCADE")


def run(make_store, body):
    async def main():
        store = make_store()
        await store.open()
        try:
            return await body(store)
        finally:
            await store.close()
    return asyncio.run(main())


async def new_user(store, email=None):
    return await store.create_user(uuid.uuid4().hex, email or f"{uuid.uuid4().hex[:8]}@test.dev", "not-a-real-hash")


async def default_owner(store):
    return (await store.get_user_by_email("owner@test.dev") or await new_user(store, "owner@test.dev")).id


async def new_session(store, sid=None, *, cwd="/w", source="cli", parent=None, owner=None):
    sid = sid or uuid.uuid4().hex[:12]
    await store.create_session(sid, owner_id=owner or await default_owner(store), source=source,
                               provider="p", model="m", system_prompt="sys", cwd=cwd,
                               parent_session_id=parent)
    return sid


def tool_round(call_id="c1", name="run_shell", args='{"command": "ls"}', result="a.txt"):
    """Exactly the dict shapes loop.record_assistant_message / run_tool_round build."""
    return [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]},
        {"role": "tool", "tool_call_id": call_id, "name": name, "content": result},
    ]


TRANSCRIPT = [{"role": "user", "content": "list the files"}, *tool_round(),
              {"role": "assistant", "content": "There is one file: a.txt"}]


# ── Contract ─────────────────────────────────────────────────────────────────

def test_round_trip_is_byte_identical(make_store):
    async def body(store):
        sid = await new_session(store)
        await store.append_messages(sid, TRANSCRIPT[:3])
        await store.append_messages(sid, TRANSCRIPT[3:])
        return await store.load_history(sid)
    loaded = run(make_store, body)
    assert json.dumps(loaded) == json.dumps(TRANSCRIPT), loaded


def test_ordering_is_by_id_across_interleaved_sessions(make_store):
    async def body(store):
        a, b = await new_session(store), await new_session(store)
        for i in range(5):
            await store.append_messages(a, [{"role": "user", "content": f"a{i}"}])
            await store.append_messages(b, [{"role": "user", "content": f"b{i}"}])
        rows = await store.get_messages(a)
        return [r.content for r in rows], [r.id for r in rows]
    contents, ids = run(make_store, body)
    assert contents == [f"a{i}" for i in range(5)]
    assert ids == sorted(ids)


def test_resume_drops_dangling_tool_call(make_store):
    async def body(store):
        sid = await new_session(store)
        # Crash mid-round: two calls requested, only one answered.
        asked = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "y", "type": "function", "function": {"name": "b", "arguments": "{}"}}]}
        await store.append_messages(sid, [{"role": "user", "content": "go"}, asked,
                                          {"role": "tool", "tool_call_id": "x", "name": "a", "content": "ok"}])
        return await store.load_history(sid), len(await store.get_messages(sid))
    history, stored = run(make_store, body)
    assert history == [{"role": "user", "content": "go"}]
    assert stored == 3          # nothing is deleted; only the resumed view is repaired


def test_counters_title_usage_and_end(make_store):
    async def body(store):
        sid = await new_session(store)
        await store.append_messages(sid, [{"role": "user", "content": "  first\n question " + "x" * 200}])
        await store.append_messages(sid, TRANSCRIPT)
        await store.add_usage(sid, 100, 20)
        await store.add_usage(sid, 5, 1)
        await store.end_session(sid, "closed")
        await store.end_session(sid, "again")          # first end wins
        return await store.get_session(sid), await store.get_session("missing")
    info, missing = run(make_store, body)
    assert missing is None
    assert info.message_count == 5 and info.tool_call_count == 1
    assert info.title.startswith("first question x") and len(info.title) == 80
    assert (info.input_tokens, info.output_tokens) == (105, 21)
    assert info.end_reason == "closed" and info.ended_at is not None
    assert (info.provider, info.model, info.cwd, info.source) == ("p", "m", "/w", "cli")


def test_list_sessions_filters(make_store):
    async def body(store):
        a = await new_session(store, cwd="/one")
        b = await new_session(store, cwd="/two")
        child = await new_session(store, cwd="/one", source="subagent", parent=a)
        await store.append_messages(a, [{"role": "user", "content": "latest"}])
        return (a, b, child,
                [s.id for s in await store.list_sessions()],
                [s.id for s in await store.list_sessions(cwd="/one")],
                {s.id for s in await store.list_sessions(include_subagents=True)},
                await store.get_session(child))
    a, b, child, all_ids, one, with_children, child_info = run(make_store, body)
    assert all_ids == [a, b]                  # most recent activity first, children hidden
    assert one == [a]
    assert with_children == {a, b, child}
    assert child_info.parent_session_id == a


def test_search_roles_and_exclusions(make_store):
    async def body(store):
        a, b = await new_session(store), await new_session(store)
        child = await new_session(store, source="subagent", parent=a)
        await store.append_messages(a, [{"role": "user", "content": "how is the zebra pipeline configured"},
                                        *tool_round(result="zebra config lives in tool output")])
        await store.append_messages(b, [{"role": "assistant", "content": "the zebra pipeline uses kafka"}])
        await store.append_messages(child, [{"role": "assistant", "content": "zebra from a subagent"}])
        return (a, b, child,
                await store.search("zebra"),
                await store.search("zebra", roles=("tool",)),
                await store.search("zebra", exclude_session_ids=[b]),
                await store.search("zebra", include_subagents=True),
                await store.search("nothing-matches-this"))
    a, b, child, default, tools, excluded, with_children, none = run(make_store, body)
    assert {(h.session_id, h.role) for h in default} == {(a, "user"), (b, "assistant")}
    assert [h.session_id for h in tools] == [a] and tools[0].role == "tool"
    assert {h.session_id for h in excluded} == {a}
    assert child in {h.session_id for h in with_children}
    assert none == []
    assert all("zebra" in h.snippet.lower() for h in default)
    assert all(h.title for h in default if h.session_id == a)


def test_messages_around(make_store):
    async def body(store):
        sid = await new_session(store)
        await store.append_messages(sid, [{"role": "user", "content": f"m{i}"} for i in range(10)])
        rows = await store.get_messages(sid)
        mid = await store.messages_around(sid, rows[5].id, window=2)
        edge = await store.messages_around(sid, rows[1].id, window=3)
        return rows, mid, edge
    rows, mid, edge = run(make_store, body)
    assert [m.content for m in mid["messages"]] == ["m3", "m4", "m5", "m6", "m7"]
    assert (mid["messages_before"], mid["messages_after"]) == (2, 2)
    assert [m.content for m in edge["messages"]][:2] == ["m0", "m1"]
    assert edge["messages_before"] == 1 and edge["messages_after"] == 3


def test_append_to_unknown_session_raises(make_store):
    async def body(store):
        with pytest.raises(KeyError):
            await store.append_messages("nope", [{"role": "user", "content": "x"}])
    run(make_store, body)


# ── Serialization ────────────────────────────────────────────────────────────

def test_serialize_helpers():
    for msg in TRANSCRIPT:
        assert row_to_message(message_to_row(msg)) == msg
    structured = message_to_row({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    assert json.loads(structured["content"]) == [{"type": "text", "text": "hi"}]
    assert repair_for_resume(list(TRANSCRIPT)) == TRANSCRIPT       # complete rounds are kept
    assert repair_for_resume(TRANSCRIPT[:2]) == TRANSCRIPT[:1]
    assert make_title("   ") is None and make_title("a\nb") == "a b"


def test_default_store_follows_env(monkeypatch):
    assert get_default_store() is None
    monkeypatch.setenv("GG_DATABASE_URL", "postgresql://u:p@127.0.0.1:1/x")
    store = get_default_store()
    assert type(store).__name__ == "PostgresSessionStore" and store.dsn.endswith("/x")


# ── Loop + Agent integration ─────────────────────────────────────────────────

SCRIPT = []

class PersistFakeTransport(ProviderTransport):
    @property
    def api_mode(self): return "persist-fake"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, messages, **kw): return messages
    def convert_tools(self, tools): return tools
    def build_kwargs(self, model, messages, tools=None, **params):
        return {"model": model, "messages": list(messages), "tools": tools}
    def call(self, client, **api_kwargs):
        system = api_kwargs["messages"][0]["content"]
        if "focused subagent" in system:
            return NormalizedResponse(content="child done", tool_calls=None, finish_reason="stop")
        return SCRIPT.pop(0)
    def normalize_response(self, response): return response

register_transport(PersistFakeTransport())
register_provider(ProviderProfile(name="persist-fake", api_mode="persist-fake", default_model="pf-1"))


def say(text):
    return NormalizedResponse(content=text, tool_calls=None, finish_reason="stop", usage=Usage(10, 2, 12))


def call(name, args, call_id="c1"):
    return NormalizedResponse(content=None, finish_reason="tool_calls", usage=Usage(20, 5, 25),
                              tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))])


class FlakyStore(InMemorySessionStore):
    """Fails the first ``failures`` appends, then behaves."""
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
    async def append_messages(self, session_id, messages):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("db went away")
        await super().append_messages(session_id, messages)


def owned(store, email=None):
    """Agent kwargs for a store plus a fresh registered owner."""
    return {"store": store, "user_id": run_sync(new_user(store, email)).id}


def roles(store, sid):
    return [m.role for m in run_sync(store.get_messages(sid))]


def public(messages):
    """History without the loop's internal bookkeeping keys (``_persisted``)."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def test_agent_persists_turn_in_order_and_resumes():
    store = InMemorySessionStore()
    SCRIPT[:] = [call("run_shell", {"command": "echo hi"}), say("it printed hi")]
    a = Agent(provider="persist-fake", **owned(store))
    r = a.run("run echo hi")
    assert r["persisted"] == 4
    assert roles(store, a.session_id) == ["user", "assistant", "tool", "assistant"]
    info = run_sync(store.get_session(a.session_id))
    assert info.title == "run echo hi" and info.tool_call_count == 1
    assert (info.input_tokens, info.output_tokens) == (30, 7)
    assert info.system_prompt == a.system_prompt and "session_search" in a.system_prompt

    b = Agent(provider="persist-fake", store=store, user_id=a.user_id, resume=a.session_id)
    assert run_sync(b.astart())
    assert b.session_id == a.session_id and b.history == a.history
    SCRIPT[:] = [say("you asked me to run echo hi")]
    b.run("what did I ask?")
    assert roles(store, a.session_id) == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    b.close()
    assert run_sync(store.get_session(a.session_id)).end_reason == "closed"


def test_resume_unknown_session_raises():
    a = Agent(provider="persist-fake", **owned(InMemorySessionStore()), resume="does-not-exist")
    with pytest.raises(ValueError):
        a.run("hi")
    with pytest.raises(RuntimeError):                       # resume without persistence
        Agent(provider="persist-fake", store=False, resume="x").run("hi")


def test_store_failure_never_fails_the_turn_and_is_retried():
    store, events = FlakyStore(failures=1), []
    SCRIPT[:] = [call("run_shell", {"command": "echo hi"}), say("done")]
    a = Agent(provider="persist-fake", **owned(store), event_callback=lambda k, p: events.append((k, p)))
    r = a.run("go")
    assert r["response"] == "done" and not r["failed"]
    assert [p["stage"] for k, p in events if k == "persist_error"] == ["append"]
    # The failed tail was retried at the next flush point: nothing lost, nothing doubled.
    assert roles(store, a.session_id) == ["user", "assistant", "tool", "assistant"]


def test_unflushed_tail_carries_over_to_the_next_turn():
    store = FlakyStore(failures=10)
    SCRIPT[:] = [say("one")]
    a = Agent(provider="persist-fake", **owned(store))
    r = a.run("first")
    assert r["persisted"] == 0 and roles(store, a.session_id) == []
    store.failures = 0
    SCRIPT[:] = [say("two")]
    a.run("second")
    assert [m.content for m in run_sync(store.get_messages(a.session_id))] == ["first", "one", "second", "two"]


def test_unreachable_store_disables_persistence():
    class DownStore(InMemorySessionStore):
        async def open(self): raise ConnectionError("refused")
    events = []
    SCRIPT[:] = [say("still works")]
    a = Agent(provider="persist-fake", **owned(DownStore()), event_callback=lambda k, p: events.append((k, p)))
    assert a.run("hi")["response"] == "still works"
    assert a.store is None and ("persist_error", "open") in [(k, p.get("stage")) for k, p in events]
    assert "session_search" not in [d["name"] for d in a.tool_definitions()]


def test_reset_starts_a_new_session_and_ends_the_old_one():
    store = InMemorySessionStore()
    SCRIPT[:] = [say("one"), say("two")]
    a = Agent(provider="persist-fake", **owned(store))
    a.run("first")
    old = a.session_id
    a.reset()
    assert a.session_id != old and a.history == []
    a.run("second")
    assert run_sync(store.get_session(old)).end_reason == "new_session"
    assert roles(store, a.session_id) == ["user", "assistant"]


def test_stateless_turn_is_not_recorded():
    store = InMemorySessionStore()
    SCRIPT[:] = [say("x")]
    a = Agent(provider="persist-fake", **owned(store))
    a.run("one-shot", keep_history=False)
    assert run_sync(store.list_sessions()) == []


def test_crash_mid_tool_round_resumes_cleanly():
    async def boom(**kw):
        raise asyncio.CancelledError()
    registry.register(name="persist_test_boom", toolset="persist-test", handler=boom,
                      schema={"name": "persist_test_boom", "parameters": {"type": "object", "properties": {}}})
    try:
        store = InMemorySessionStore()
        SCRIPT[:] = [call("persist_test_boom", {})]
        a = Agent(provider="persist-fake", **owned(store))
        # Through run_sync the task's cancellation surfaces as the future's.
        with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
            a.run("do the risky thing")
        # user + assistant(tool_calls) were durable before the tool ran...
        assert roles(store, a.session_id) == ["user", "assistant"]
        # ...and resume drops the unanswered call so the next request is valid.
        b = Agent(provider="persist-fake", store=store, user_id=a.user_id, resume=a.session_id)
        run_sync(b.astart())
        # (resumed messages carry the loop's durability marker; compare the wire fields)
        assert public(b.history) == [{"role": "user", "content": "do the risky thing"}]
    finally:
        registry._tools.pop("persist_test_boom", None)


def test_subagents_share_the_store_and_link_to_the_parent():
    store = InMemorySessionStore()
    SCRIPT[:] = [call("delegate_task", {"goal": "count files"}, call_id="d1"), say("combined")]
    a = Agent(provider="persist-fake", **owned(store))
    assert a.run("delegate it")["response"] == "combined"
    children = [s for s in run_sync(store.list_sessions(include_subagents=True)) if s.id != a.session_id]
    assert len(children) == 1
    child = children[0]
    assert child.source == "subagent" and child.parent_session_id == a.session_id
    assert child.end_reason == "closed"
    assert roles(store, child.id) == ["user", "assistant"]
    assert [s.id for s in run_sync(store.list_sessions())] == [a.session_id]


def test_no_store_means_no_change():
    SCRIPT[:] = [say("plain")]
    a = Agent(provider="persist-fake")
    assert a.store is None and "session_search" not in a.system_prompt
    assert "session_search" not in [d["name"] for d in a.tool_definitions()]
    a.run("hi")
    a.reset()
    assert a.history == []


# ── Users & ownership ───────────────────────────────────────────────────────


def test_register_and_authenticate(make_store):
    async def body(store):
        user = await register_user(store, "  Ana@Example.COM ", "correct horse")
        with pytest.raises(ValueError, match="already registered"):
            await register_user(store, "ana@example.com", "another password")
        with pytest.raises(ValueError):
            await register_user(store, "not-an-email", "long enough pw")
        with pytest.raises(ValueError, match="at least"):
            await register_user(store, "bob@example.com", "short")
        return (user, await store.get_user(user.id),
                await authenticate(store, "ANA@example.com", "correct horse"),
                await authenticate(store, "ana@example.com", "wrong password"),
                await authenticate(store, "nobody@example.com", "correct horse"))
    user, fetched, ok, wrong, unknown = run(make_store, body)
    assert user.email == "ana@example.com" and len(user.id) == 32
    assert fetched.email == user.email and "correct horse" not in fetched.password_hash
    assert fetched.password_hash.startswith("scrypt$")
    assert ok is not None and ok.id == user.id
    assert wrong is None and unknown is None


def test_password_hashing():
    h1, h2 = hash_password("s3cret-pass"), hash_password("s3cret-pass")
    assert h1 != h2                                    # salted
    assert verify_password("s3cret-pass", h1) and not verify_password("s3cret-pasS", h1)
    assert not verify_password("x", "garbage") and not verify_password("x", "")
    assert h1 not in repr(UserInfo("id", "e@x.dev", h1))    # never lands in logs or tracebacks


def test_sessions_are_scoped_to_their_owner(make_store):
    async def body(store):
        ana, bob = await new_user(store), await new_user(store)
        a = await new_session(store, owner=ana.id)
        b = await new_session(store, owner=bob.id)
        await store.append_messages(a, [{"role": "user", "content": "ana talks about llamas"}])
        await store.append_messages(b, [{"role": "user", "content": "bob talks about llamas"}])
        with pytest.raises(KeyError):
            await new_session(store, owner="no-such-user")
        return (ana.id, a, b, await store.get_session(a),
                [s.id for s in await store.list_sessions(owner_id=ana.id)],
                {h.session_id for h in await store.search("llamas", owner_id=bob.id)},
                {h.session_id for h in await store.search("llamas")})
    ana_id, a, b, info, ana_list, bob_hits, all_hits = run(make_store, body)
    assert info.owner_id == ana_id
    assert ana_list == [a] and bob_hits == {b} and all_hits == {a, b}


def test_agent_without_user_saves_nothing():
    store, events = InMemorySessionStore(), []
    SCRIPT[:] = [say("hi")]
    a = Agent(provider="persist-fake", store=store, event_callback=lambda k, p: events.append((k, p)))
    assert a.store is None and "session_search" not in a.system_prompt
    a.run("hello")
    assert run_sync(store.list_sessions()) == []
    assert [p["stage"] for k, p in events if k == "persist_error"] == ["owner"]


def test_agent_with_unknown_user_saves_nothing():
    store, events = InMemorySessionStore(), []
    SCRIPT[:] = [say("hi")]
    a = Agent(provider="persist-fake", store=store, user_id="deleted-user",
              event_callback=lambda k, p: events.append((k, p)))
    a.run("hello")
    assert a.store is None and run_sync(store.list_sessions()) == []
    assert ("persist_error", "owner") in [(k, p.get("stage")) for k, p in events]


def test_sessions_record_their_owner_and_others_cannot_resume_them():
    store = InMemorySessionStore()
    SCRIPT[:] = [call("delegate_task", {"goal": "count files"}, call_id="d1"), say("done")]
    a = Agent(provider="persist-fake", **owned(store))
    a.run("delegate it")
    sessions = run_sync(store.list_sessions(include_subagents=True))
    assert len(sessions) == 2 and {s.owner_id for s in sessions} == {a.user_id}   # the child too

    intruder = Agent(provider="persist-fake", **owned(store), resume=a.session_id)
    with pytest.raises(ValueError, match="No such session"):
        run_sync(intruder.astart())
    owner_again = Agent(provider="persist-fake", store=store, user_id=a.user_id, resume=a.session_id)
    assert run_sync(owner_again.astart()) and owner_again.history == a.history
