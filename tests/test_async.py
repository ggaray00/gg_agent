"""Async core: sync wrappers, concurrency, async tools, cancellation (no network)."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gg_agent import Agent, ProviderProfile, register_provider, register_transport
from gg_agent.aio import in_async_context, run_sync
from gg_agent.tools.registry import registry
from gg_agent.transports.base import ProviderTransport
from gg_agent.transports.types import NormalizedResponse, ToolCall

DELAY = 0.20      # per fake API call

class SlowAsyncTransport(ProviderTransport):
    """An async transport that takes real (small) wall-clock time."""
    @property
    def api_mode(self): return "slow"
    def build_client(self, *, api_key, base_url, profile): return object()
    def convert_messages(self, m, **kw): return m
    def convert_tools(self, t): return t
    def build_kwargs(self, model, messages, tools=None, **p):
        return {"model": model, "messages": messages, "tools": tools}
    async def call(self, client, **kw):
        await asyncio.sleep(DELAY)
        return NormalizedResponse(content="done", tool_calls=None, finish_reason="stop")
    def normalize_response(self, r): return r

register_transport(SlowAsyncTransport())
register_provider(ProviderProfile(name="slow", api_mode="slow", default_model="s-1"))


# sync wrapper still works, and is what the old API promised
def test_1_sync_wrapper_runs_an_async_core():
    a = Agent(provider="slow")
    r = a.run("hi")
    assert r["response"] == "done" and r["api_calls"] == 1
    assert a.ask("again") == "done"
    a.close()
    print("✓ 1 sync run/ask/close over the async core")


# awaiting directly is the native path
def test_2_async_api():
    async def go():
        async with Agent(provider="slow") as a:
            assert (await a.aask("hi")) == "done"
            r = await a.arun("hi again")
            assert r["exit_reason"] == "final_response"
            return True
    assert asyncio.run(go())
    print("✓ 2 arun/aask/async with")


# N agents concurrently cost ~1 delay, not N
def test_3_agents_run_concurrently():
    async def go():
        agents = [Agent(provider="slow") for _ in range(5)]
        started = time.time()
        answers = await asyncio.gather(*(a.aask("q") for a in agents))
        elapsed = time.time() - started
        await asyncio.gather(*(a.aclose() for a in agents))
        return answers, elapsed
    answers, elapsed = asyncio.run(go())
    assert answers == ["done"] * 5
    # Sequential would be 5*DELAY; allow generous slack for a loaded machine.
    assert elapsed < DELAY * 3, f"5 agents took {elapsed:.2f}s — they serialized"
    print(f"✓ 3 five agents in {elapsed:.2f}s (sequential would be {DELAY*5:.2f}s)")


# calling the sync wrapper from inside a loop raises instead of deadlocking
def test_4_sync_inside_loop_raises():
    async def go():
        assert in_async_context()
        a = Agent(provider="slow")
        try:
            a.run("boom")
        except RuntimeError as exc:
            assert "already running event loop" in str(exc) or "running event loop" in str(exc)
            await a.aclose()
            return True
        return False
    assert asyncio.run(go()), "sync run() inside a loop must raise"
    print("✓ 4 sync-inside-async raises, not deadlocks")


# async and sync handlers both dispatch; sync ones don't block the loop
def test_5_registry_dispatches_both_handler_kinds():
    async def ahandler(**kw): return "async-result"
    def shandler(**kw): time.sleep(0.05); return "sync-result"

    for name, fn in (("t_async", ahandler), ("t_sync", shandler)):
        registry.register(name=name, toolset="testing", handler=fn, override=True,
                          schema={"name": name, "parameters": {"type": "object", "properties": {}}})
    assert registry.get("t_async").is_async and not registry.get("t_sync").is_async

    async def go():
        # A blocking sync tool must not stop siblings from progressing.
        started = time.time()
        results = await asyncio.gather(*(registry.dispatch(n, {}) for n in ("t_sync",) * 4))
        return results, time.time() - started
    results, elapsed = asyncio.run(go())
    assert results == ["sync-result"] * 4
    assert elapsed < 0.15, f"sync tools serialized ({elapsed:.2f}s) — not offloaded to threads"
    assert run_sync(registry.dispatch("t_async", {})) == "async-result"
    print(f"✓ 5 async + sync handlers, 4 blocking tools in {elapsed:.2f}s")


# a tool round runs its calls in parallel and keeps reply order
def test_6_parallel_tool_round_preserves_order():
    order = []
    async def slow_tool(tag=None, **kw):
        await asyncio.sleep(0.10 if tag == "a" else 0.01)
        order.append(tag)
        return f"ran-{tag}"
    registry.register(name="ordered", toolset="testing", handler=slow_tool, override=True,
                      schema={"name": "ordered", "parameters": {
                          "type": "object", "properties": {"tag": {"type": "string"}}}})

    class T(SlowAsyncTransport):
        @property
        def api_mode(self): return "ordered"
        async def call(self, client, **kw):
            if any(m.get("role") == "tool" for m in kw["messages"]):
                return NormalizedResponse(content="finished", tool_calls=None, finish_reason="stop")
            return NormalizedResponse(content=None, finish_reason="tool_calls", tool_calls=[
                ToolCall(id="1", name="ordered", arguments='{"tag": "a"}'),
                ToolCall(id="2", name="ordered", arguments='{"tag": "b"}'),
            ])
    register_transport(T())
    register_provider(ProviderProfile(name="ordered", api_mode="ordered", default_model="o"))

    a = Agent(provider="ordered")
    r = a.run("go")
    a.close()
    tool_msgs = [m for m in r["history"] if m.get("role") == "tool"]
    # b finishes first (it is faster) but the replies stay in request order.
    assert order == ["b", "a"], order
    assert [m["tool_call_id"] for m in tool_msgs] == ["1", "2"], tool_msgs
    assert [m["content"] for m in tool_msgs] == ["ran-a", "ran-b"], tool_msgs
    print("✓ 6 tools run in parallel, replies keep request order")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
