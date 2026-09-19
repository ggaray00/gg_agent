"""session_search tool: the four modes, exclusions, dedup, adaptive detail (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.aio import run_sync
from gg_agent.persistence import InMemorySessionStore
from gg_agent.tools.delegate_tool import _build_child
from gg_agent.tools.session_tools import session_search
from gg_agent.transports.base import ProviderTransport


class NoCallTransport(ProviderTransport):
    @property
    def api_mode(self): return "search-fake"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, m, **kw): return m
    def convert_tools(self, t): return t
    def build_kwargs(self, model, messages, tools=None, **p): return {}
    def call(self, client, **kw): raise AssertionError("the tool must not call the model")
    def normalize_response(self, r): return r

register_transport(NoCallTransport())
register_provider(ProviderProfile(name="search-fake", api_mode="search-fake", default_model="sf"))


def seed():
    """Three past sessions, a subagent one, the agent's own current session — and
    another user's session that must never surface."""
    store = InMemorySessionStore()
    me = run_sync(store.create_user("me", "me@test.dev", "h"))
    other = run_sync(store.create_user("other", "other@test.dev", "h"))
    agent = Agent(provider="search-fake", store=store, user_id=me.id)

    async def build():
        async def session(sid, messages, source="cli", parent=None, owner=me.id):
            await store.create_session(sid, owner_id=owner, source=source, provider="p", model="m",
                                       system_prompt="s", cwd="/w", parent_session_id=parent)
            await store.append_messages(sid, messages)

        await session("pooling", [
            {"role": "user", "content": "how should we size the postgres pool?"},
            {"role": "assistant", "content": "use min_size 1 and max_size 5 for the postgres pool"},
            {"role": "user", "content": "and the postgres timeout?"},
            {"role": "assistant", "content": "five seconds"},
        ])
        await session("long", [{"role": "user" if i % 2 == 0 else "assistant", "content": f"long msg {i}"}
                               for i in range(40)] + [{"role": "user", "content": "postgres at the end"}])
        await session("other", [{"role": "user", "content": "unrelated chat about lunch " + "y" * 5000}])
        await session("child", [{"role": "assistant", "content": "postgres from a subagent"}],
                      source="subagent", parent="pooling")
        await session("theirs", [{"role": "user", "content": "postgres secrets of another user"}], owner=other.id)
        await agent.astart()
        await agent._ensure_session()
        await store.append_messages(agent.session_id, [{"role": "user", "content": "postgres right now"}])
    run_sync(build())
    return store, agent


def search(agent, **kw):
    return run_sync(session_search(parent_agent=agent, **kw))


def test_discover_dedups_excludes_and_hydrates_only_the_top_session():
    _, agent = seed()
    r = search(agent, query="postgres", limit=5)
    assert r["mode"] == "discover"
    ids = [e["session_id"] for e in r["results"]]
    assert sorted(ids) == ["long", "pooling"]            # current session + subagent excluded, one per session
    assert "context" in r["results"][0] and "context" not in r["results"][1]
    top = r["results"][0]
    assert top["match_message_id"] in [m["id"] for m in top["context"]]
    assert "around_message_id" in r["hint"]


def test_discover_respects_limit_and_roles():
    _, agent = seed()
    assert len(search(agent, query="postgres", limit=1)["results"]) == 1
    only_assistant = search(agent, query="postgres", role_filter=["assistant"])
    assert [e["session_id"] for e in only_assistant["results"]] == ["pooling"]
    assert only_assistant["results"][0]["match_role"] == "assistant"


def test_zero_results_explain_the_syntax():
    _, agent = seed()
    r = search(agent, query="kubernetes")
    assert r["results"] == [] and '"quoted phrase"' in r["message"] and "-term" in r["message"]


def test_scroll_window_and_edges():
    store, agent = seed()
    rows = run_sync(store.get_messages("long"))
    r = search(agent, session_id="long", around_message_id=rows[10].id, window=3)
    assert r["mode"] == "scroll"
    assert [m["content"] for m in r["messages"]] == [f"long msg {i}" for i in range(7, 14)]
    assert r["messages_before"] == 3 and r["messages_after"] == 3
    edge = search(agent, session_id="long", around_message_id=rows[0].id, window=99)   # clamped to 20
    assert edge["messages_before"] == 0 and edge["messages_after"] == 20
    missing = search(agent, session_id="nope", around_message_id=1)
    assert missing["messages"] == [] and "No messages" in missing["message"]


def test_read_gives_head_and_tail_of_long_sessions():
    _, agent = seed()
    r = search(agent, session_id="long")
    assert r["mode"] == "read" and r["title"] == "long msg 0"
    assert len(r["messages"]) == 20 and len(r["tail"]) == 10 and r["omitted"] == 11
    assert r["tail"][-1]["content"] == "postgres at the end"
    short = search(agent, session_id="pooling")
    assert len(short["messages"]) == 4 and "tail" not in short
    assert "error" in search(agent, session_id="nope")


def test_long_messages_are_capped():
    _, agent = seed()
    r = search(agent, session_id="other")
    assert len(r["messages"][0]["content"]) < 4_100 and "truncated" in r["messages"][0]["content"]


def test_browse_lists_recent_sessions_without_subagents():
    _, agent = seed()
    r = search(agent)
    assert r["mode"] == "browse"
    ids = [s["session_id"] for s in r["sessions"]]
    assert "child" not in ids and {"pooling", "long", "other", agent.session_id} <= set(ids)
    assert [s["current"] for s in r["sessions"] if s["session_id"] == agent.session_id] == [True]


def test_tool_availability_follows_the_store():
    _, agent = seed()
    assert "session_search" in [d["name"] for d in agent.tool_definitions()]
    assert "use session_search to recall it" in agent.system_prompt

    plain = Agent(provider="search-fake")
    assert "session_search" not in [d["name"] for d in plain.tool_definitions()]
    assert "session_search" not in plain.system_prompt
    assert "error" in run_sync(session_search(query="x", parent_agent=plain))

    blocked = Agent(provider="search-fake", store=agent.store, user_id=agent.user_id,
                    blocked_tools={"session_search"})
    assert "session_search" not in blocked.system_prompt

    child = _build_child(agent, {"goal": "g"}, 0)
    assert child.store is agent.store and child.source == "subagent" and child.user_id == agent.user_id
    assert "session_search" not in [d["name"] for d in child.tool_definitions()]


def test_other_users_sessions_are_invisible():
    _, agent = seed()
    assert "theirs" not in [e["session_id"] for e in search(agent, query="postgres", limit=10)["results"]]
    assert search(agent, query="secrets")["results"] == []
    assert "theirs" not in [s["session_id"] for s in search(agent)["sessions"]]
    assert "error" in search(agent, session_id="theirs")                       # reads as missing
    assert search(agent, session_id="theirs", around_message_id=1)["messages"] == []
