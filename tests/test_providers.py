"""Provider profiles ported from hermes-agent, and the transports they need (no network)."""
import asyncio
import base64
import json
import sys
import textwrap
import time
from types import SimpleNamespace as NS

import pytest

from gg_agent.providers import get_provider_profile, iter_configured, list_providers
from gg_agent.reasoning_effort import parse_reasoning
from gg_agent.transports import get_transport
from gg_agent.transports.chat_completions import ChatCompletionsTransport

CC = ChatCompletionsTransport()
MSG = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

# Every env var any profile reads, so the host's real keys can't leak into a test.
_ALL_VARS = {v for p in list_providers() for v in p.env_vars} | {
    "COPILOT_GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_WEB_IDENTITY_TOKEN_FILE", "VERTEX_PROJECT_ID", "VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS",
    "GG_REASONING", "GG_PROVIDER"}


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    import gg_agent.providers.copilot_auth as ca
    monkeypatch.setattr(ca, "_CRED_FILES", (str(tmp_path / "absent.json"),))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    return monkeypatch


def kwargs_for(provider, model, reasoning=None, **extra):
    return CC.build_kwargs(model, MSG, profile=get_provider_profile(provider),
                           reasoning_config=reasoning, **extra)


# ── Registry ─────────────────────────────────────────────────────────────────

def test_every_hermes_provider_is_registered():
    names = {p.name for p in list_providers()}
    hermes = {
        "actual", "ai-gateway", "alibaba", "alibaba-cn", "alibaba-token-plan", "alibaba-token-plan-cn",
        "alibaba-coding-plan", "alibaba-coding-plan-cn", "anthropic", "arcee", "azure-foundry", "bedrock",
        "commandcode", "commandcode-anthropic", "copilot", "copilot-acp", "custom", "deepinfra", "deepseek",
        "fireworks", "gemini", "gmi", "huggingface", "kilocode", "kimi-coding", "kimi-coding-cn", "meta-ai",
        "minimax", "minimax-cn", "nebius-token-factory", "nous", "novita", "nvidia", "ollama-cloud",
        "openai-codex", "opencode-free", "opencode-zen", "opencode-go", "openrouter", "qwen-oauth", "router",
        "stepfun", "upstage", "vertex", "xai", "xiaomi", "zai",
    }
    assert hermes <= names, hermes - names
    assert {"openai", "groq", "ollama"} <= names, "gg-only providers are kept"


def test_aliases_resolve_and_never_collide():
    seen = {}
    for p in list_providers():
        for alias in p.aliases:
            assert alias not in seen, f"{alias} claimed by {seen[alias]} and {p.name}"
            assert alias not in {q.name for q in list_providers()}, f"alias {alias} shadows a provider"
            seen[alias] = p.name
            assert get_provider_profile(alias) is p
    assert get_provider_profile("grok").name == "xai"
    assert get_provider_profile("GLM").name == "zai"


def test_every_api_mode_has_a_transport():
    for p in list_providers():
        get_transport(p.api_mode)


def test_base_url_vars_are_not_keys(clean_env):
    p = get_provider_profile("novita")
    clean_env.setenv("NOVITA_BASE_URL", "https://proxy.example.com/v1/")
    assert p.resolve_api_key() == "" and not p.has_credentials()
    clean_env.setenv("NOVITA_API_KEY", "k")
    assert p.resolve_credentials() == ("k", "https://proxy.example.com/v1")


def test_auto_detect_keeps_the_original_order(clean_env):
    clean_env.setenv("DASHSCOPE_API_KEY", "x")
    clean_env.setenv("OPENAI_API_KEY", "y")
    from gg_agent.agent import resolve_provider
    assert resolve_provider().name == "openai", "a new provider must not steal an existing setup"
    clean_env.delenv("OPENAI_API_KEY")
    assert resolve_provider().name == "alibaba"


def test_keyless_providers_are_never_auto_picked(clean_env):
    from gg_agent.agent import resolve_provider
    assert get_provider_profile("ollama") in list(iter_configured())
    with pytest.raises(RuntimeError):
        resolve_provider()


def test_default_model_falls_back_to_the_catalog():
    assert get_provider_profile("huggingface").default_model == "Qwen/Qwen3.5-72B-Instruct"


# ── Reasoning translation (chat_completions) ────────────────────────────────

def test_parse_reasoning():
    assert parse_reasoning(None) is None
    assert parse_reasoning("off") == {"enabled": False}
    assert parse_reasoning("High") == {"enabled": True, "effort": "high"}


def test_no_reasoning_fields_unless_asked():
    kw = kwargs_for("openrouter", "openai/gpt-5.4")
    assert "reasoning" not in kw.get("extra_body", {})


def test_openrouter_reasoning_and_mandatory_claude():
    kw = kwargs_for("openrouter", "openai/gpt-5.4", {"enabled": True, "effort": "low"})
    assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "low"}
    claude = kwargs_for("openrouter", "anthropic/claude-sonnet-5", {"enabled": True, "effort": "high"})
    assert "reasoning" not in claude.get("extra_body", {}) and claude["verbosity"] == "high"


def test_deepseek_v4_always_sets_thinking():
    kw = kwargs_for("deepseek", "deepseek-v4-pro", {"enabled": True, "effort": "xhigh"})
    assert kw["extra_body"]["thinking"] == {"type": "enabled"} and kw["reasoning_effort"] == "max"
    off = kwargs_for("deepseek", "deepseek-v4-pro", {"enabled": False})
    assert off["extra_body"]["thinking"] == {"type": "disabled"}
    assert "extra_body" not in kwargs_for("deepseek", "deepseek-v3.2"), "V3 untouched"


def test_kimi_thinking_xor_effort_and_no_temperature():
    kw = kwargs_for("kimi", "kimi-k3", {"enabled": True, "effort": "medium"}, temperature=0.7)
    assert kw["reasoning_effort"] == "high" and "thinking" not in kw.get("extra_body", {})
    assert "temperature" not in kw, "Kimi manages temperature server-side"


def test_zai_glm_5_2_floor():
    kw = kwargs_for("zai", "glm-5.2", {"enabled": True, "effort": "low"})
    assert kw["extra_body"]["thinking"] == {"type": "enabled"} and kw["reasoning_effort"] == "high"


def test_upstage_defaults_reasoning_on():
    assert kwargs_for("upstage", "solar-pro3")["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in kwargs_for("upstage", "solar-mini")


def test_gemini_nested_extra_body():
    kw = kwargs_for("gemini", "gemini-3-flash", {"enabled": True, "effort": "xhigh"})
    assert kw["extra_body"]["extra_body"]["google"]["thinking_config"] == {
        "include_thoughts": True, "thinking_level": "high"}
    assert "extra_body" not in kwargs_for("gemini", "gemma-3", {"enabled": True, "effort": "high"})


def test_nvidia_strips_tool_names():
    msgs = [{"role": "tool", "tool_call_id": "1", "name": "f", "content": "ok"}]
    kw = CC.build_kwargs("m", msgs, profile=get_provider_profile("nvidia"))
    assert "name" not in kw["messages"][0] and kw["max_tokens"] == 16384


def test_qwen_blocks_and_cache_marker():
    kw = kwargs_for("qwen", "qwen3-coder")
    assert kw["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kw["extra_body"]["vl_high_resolution_images"] is True


def test_opencode_go_caps_max_tokens_per_model():
    p = get_provider_profile("opencode-go")
    assert p.get_max_tokens("mimo-v2.5-pro") == 131072 and p.get_max_tokens("glm-5") is None


def test_custom_ollama_think_off():
    kw = kwargs_for("custom", "qwen3", {"enabled": False}, base_url="http://localhost:11434/v1")
    assert kw["reasoning_effort"] == "none" and kw["extra_body"]["think"] is False


# ── Responses transport ─────────────────────────────────────────────────────

RT = get_transport("codex_responses")
CONVO = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
]


def test_responses_conversion():
    kw = RT.build_kwargs("grok-4.6", CONVO, tools=[{"name": "ls", "description": "d", "parameters": {}}],
                         profile=get_provider_profile("xai"), session_id="s", max_tokens=100,
                         reasoning_config={"enabled": True, "effort": "max"})
    assert kw["instructions"] == "be brief" and kw["store"] is False
    assert [i.get("type", i.get("role")) for i in kw["input"]] == ["user", "function_call", "function_call_output"]
    assert kw["tools"][0]["name"] == "ls" and kw["tools"][0]["type"] == "function"
    assert kw["reasoning"] == {"effort": "xhigh"}, "clamped to Grok 4.6's vocabulary"
    assert kw["extra_headers"] == {"x-grok-conv-id": "s"} and kw["max_output_tokens"] == 100


def test_responses_meta_never_sends_none():
    kw = RT.build_kwargs("muse-spark-1.2", CONVO, profile=get_provider_profile("meta-ai"),
                         reasoning_config={"enabled": False})
    assert "reasoning" not in kw


def test_responses_codex_controls():
    kw = RT.build_kwargs("gpt-5.5", CONVO, profile=get_provider_profile("openai-codex"), max_tokens=100)
    assert kw["_stream_only"] is True and "max_output_tokens" not in kw
    assert "_omit_max_output_tokens" not in kw


def test_responses_normalize():
    resp = NS(status="completed", usage=NS(input_tokens=10, output_tokens=5, total_tokens=15,
                                           input_tokens_details=NS(cached_tokens=4)), output=[
        NS(type="reasoning", summary=[NS(text="thinking")], content=None),
        NS(type="message", content=[NS(type="output_text", text="hello")]),
        NS(type="function_call", call_id="c9", name="ls", arguments='{"a": 1}'),
    ])
    r = RT.normalize_response(resp)
    assert r.content == "hello" and r.reasoning == "thinking" and r.finish_reason == "tool_calls"
    assert r.tool_calls[0].id == "c9" and r.usage.cached_tokens == 4 and r.usage.prompt_tokens == 10


def test_responses_stream_uses_done_items_when_final_output_is_empty():
    events = [
        NS(type="response.output_text.delta", delta="hel"),
        NS(type="response.output_text.delta", delta="lo"),
        NS(type="response.output_item.done", item=NS(type="message", content=[NS(type="output_text", text="hello")])),
        NS(type="response.completed", response=NS(status="completed", output=[], usage=None)),
    ]

    class Stream:
        def __aiter__(self):
            async def gen():
                for e in events:
                    yield e
            return gen()

    async def create(**kw):
        assert kw["stream"] is True and "_stream_only" not in kw
        return Stream()

    client = NS(responses=NS(create=create))
    seen = []
    from gg_agent.transports.streaming import StreamHooks
    r = asyncio.run(RT.call_stream(client, StreamHooks(on_text=seen.append), _stream_only=True, model="m"))
    assert "".join(seen) == "hello" and r.content == "hello" and r.finish_reason == "stop"
    # call() on a stream-only backend streams too
    r2 = asyncio.run(RT.call(client, _stream_only=True, model="m"))
    assert r2.content == "hello"


# ── Bedrock transport ───────────────────────────────────────────────────────

def test_bedrock_conversion_and_normalize():
    bt = get_transport("bedrock_converse")
    kw = bt.build_kwargs("us.anthropic.claude-sonnet-4-5", CONVO + [{"role": "user", "content": "thanks"}],
                         tools=[{"name": "ls", "description": "d", "parameters": {"type": "object"}}],
                         profile=get_provider_profile("bedrock"), session_id="x")
    assert kw["system"][0] == {"text": "be brief"} and "cachePoint" in kw["system"][1]
    roles = [m["role"] for m in kw["messages"]]
    assert roles == ["user", "assistant", "user"], "tool result + next user turn fold together"
    assert kw["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "c1"
    assert kw["inferenceConfig"]["maxTokens"] == 8192 and "session_id" not in kw
    r = bt.normalize_response({"stopReason": "tool_use", "usage": {"inputTokens": 3, "outputTokens": 2,
                                                                   "cacheReadInputTokens": 7},
                               "output": {"message": {"content": [
                                   {"text": "ok"}, {"toolUse": {"toolUseId": "t", "name": "ls", "input": {}}}]}}})
    assert r.finish_reason == "tool_calls" and r.usage.prompt_tokens == 10 and r.tool_calls[0].name == "ls"


def test_bedrock_region_from_env(clean_env):
    clean_env.setenv("AWS_REGION", "eu-west-1")
    _, base = get_provider_profile("bedrock").resolve_credentials()
    from gg_agent.transports.bedrock import _region_from_base_url
    assert _region_from_base_url(base) == "eu-west-1"


# ── Anthropic transport auth ────────────────────────────────────────────────

def test_anthropic_bearer_profiles(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-not-leak")
    at = get_transport("anthropic_messages")
    client = at.build_client(api_key="mm-key", base_url="https://api.minimax.io/anthropic/v1",
                             profile=get_provider_profile("minimax"))
    assert client.api_key is None and client.auth_token == "mm-key"
    assert str(client.base_url).rstrip("/") == "https://api.minimax.io/anthropic"
    plain = at.build_client(api_key="sk-ant-api03-x", base_url="", profile=get_provider_profile("anthropic"))
    assert plain.api_key == "sk-ant-api03-x"
    kw = at.build_kwargs("claude-sonnet-5", MSG, profile=get_provider_profile("anthropic"),
                         reasoning_config={"enabled": True}, session_id="s", base_url="b")
    assert not {"reasoning_config", "session_id", "base_url"} & set(kw)


# ── OpenAI Codex auth ───────────────────────────────────────────────────────

def _jwt(claims):
    enc = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"h.{enc}.s"


def test_codex_reads_the_cli_store(clean_env, tmp_path):
    p = get_provider_profile("codex")
    assert not p.has_credentials()
    token = _jwt({"exp": time.time() + 3600, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})
    (tmp_path / "codex").mkdir()
    (tmp_path / "codex" / "auth.json").write_text(json.dumps({"tokens": {"access_token": token}}))
    assert p.has_credentials() and p.resolve_credentials() == (token, "https://chatgpt.com/backend-api/codex")
    assert p.client_headers(token)["ChatGPT-Account-ID"] == "acct-1"


# ── Copilot ACP ─────────────────────────────────────────────────────────────

FAKE_ACP = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        msg = json.loads(line)
        m, i = msg.get("method"), msg.get("id")
        if m == "initialize":
            out = {"protocolVersion": 1}
        elif m == "session/new":
            out = {"sessionId": "S"}
        elif m == "session/prompt":
            text = msg["params"]["prompt"][0]["text"]
            reply = ('<tool_call>{"id": "t1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}</tool_call>'
                     if "Tool:" not in text else "all done")
            for chunk in (reply[:5], reply[5:]):
                print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": chunk}}}}),
                    flush=True)
            out = {"stopReason": "end_turn"}
        else:
            out = {}
        print(json.dumps({"jsonrpc": "2.0", "id": i, "result": out}), flush=True)
''')


def test_copilot_acp_round_trip(tmp_path):
    script = tmp_path / "fake_acp.py"
    script.write_text(FAKE_ACP)
    from gg_agent.providers.copilot_acp_client import CopilotACPClient
    client = CopilotACPClient(command=sys.executable, args=[str(script)], cwd=str(tmp_path))
    kw = CC.build_kwargs("copilot-acp", MSG, tools=[{"name": "ls", "description": "d", "parameters": {}}],
                         profile=get_provider_profile("copilot-acp"))
    r = CC.normalize_response(asyncio.run(CC.call(client, **kw)))
    assert r.finish_reason == "tool_calls" and r.tool_calls[0].name == "ls" and not r.content
    follow = MSG + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "a.py"}]
    r2 = CC.normalize_response(asyncio.run(CC.call(client, **CC.build_kwargs("copilot-acp", follow))))
    assert r2.content == "all done" and r2.finish_reason == "stop"


def test_copilot_acp_profile_builds_its_own_client():
    from gg_agent.providers.copilot_acp_client import CopilotACPClient
    client = CC.build_client(api_key="", base_url="acp://copilot", profile=get_provider_profile("copilot-acp"))
    assert isinstance(client, CopilotACPClient) and client.args == ["--acp", "--stdio"]


# ── Agent plumbing ──────────────────────────────────────────────────────────

def test_agent_reasoning_reaches_the_request(clean_env):
    clean_env.setenv("GG_REASONING", "high")
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    from gg_agent import Agent
    from gg_agent.loop import LoopState, assemble_request
    agent = Agent(provider="openrouter", model="openai/gpt-5.4", store=False, mcp=False)
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    state = LoopState(messages=list(MSG))
    assemble_request(agent, state)
    assert state.api_kwargs["extra_body"]["reasoning"]["effort"] == "high"
    assert state.api_kwargs["extra_body"]["session_id"] == agent.session_id
