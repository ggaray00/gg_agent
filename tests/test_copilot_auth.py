"""Copilot auth tests — offline. No network, no reading the real credential store."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gg_agent.providers.copilot_auth as ca  # noqa: E402
from gg_agent.providers import get_provider_profile  # noqa: E402


def _clear_caches():
    ca._jwt_cache.clear()
    ca._failure_cache.clear()


# Classic PATs are rejected with a useful message, not an opaque 401 later
def test_1_token_validation():
    assert ca.validate_github_token("ghu_abc")[0]
    assert ca.validate_github_token("gho_abc")[0]
    assert ca.validate_github_token("github_pat_abc")[0]
    assert not ca.validate_github_token("")[0]
    ok, msg = ca.validate_github_token("ghp_classic")
    assert not ok and "Classic Personal Access Tokens" in msg


# The on-disk store is walked structurally, so a schema change doesn't break it
def test_2_oauth_token_extraction():
    vscode = {"github.com:Iv1.b507a08c87ecfe98": {"user": "me", "oauth_token": "ghu_fromapps"}}
    assert ca._oauth_token_from_blob(vscode) == "ghu_fromapps"
    hosts = {"github.com": {"user": "me", "oauth_token": "gho_fromhosts"}}
    assert ca._oauth_token_from_blob(hosts) == "gho_fromhosts"
    cli = {"copilotTokens": {"github.com": "ghu_nested"}}
    assert ca._oauth_token_from_blob(cli) == "ghu_nested"
    assert ca._oauth_token_from_blob({"user": "me"}) == ""
    assert ca._oauth_token_from_blob({"k": "not-a-token"}) == ""


def test_3_disk_store_discovery(tmp_path, monkeypatch):
    store = tmp_path / "apps.json"
    store.write_text(json.dumps(
        {"github.com:Iv1.b507a08c87ecfe98": {"user": "me", "oauth_token": "ghu_ondisk"}}))
    monkeypatch.setattr(ca, "_CRED_FILES", (str(store),))
    for var in ca.COPILOT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ca, "_token_from_gh_cli", lambda: ("", ""))
    token, source = ca.resolve_github_token()
    assert token == "ghu_ondisk" and source == str(store)


# An explicit env var beats the on-disk store
def test_4_env_precedence(tmp_path, monkeypatch):
    store = tmp_path / "apps.json"
    store.write_text(json.dumps({"x": {"oauth_token": "ghu_ondisk"}}))
    monkeypatch.setattr(ca, "_CRED_FILES", (str(store),))
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_fromenv")
    assert ca.resolve_github_token() == ("ghu_fromenv", "COPILOT_GITHUB_TOKEN")
    # A classic PAT in the env is skipped, not used
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_classic")
    monkeypatch.setattr(ca, "_token_from_gh_cli", lambda: ("", ""))
    assert ca.resolve_github_token()[0] == "ghu_ondisk"


# Enterprise/proxied accounts are NOT on the public host
def test_5_base_url_derivation():
    assert ca._derive_base_url("tid=abc;proxy-ep=proxy.individual.githubcopilot.com;x=1") \
        == "https://api.individual.githubcopilot.com"
    assert ca._derive_base_url("tid=abc;proxy-ep=https://proxy.enterprise.example.com/") \
        == "https://api.enterprise.example.com"
    assert ca._derive_base_url("tid=abc;no-proxy-field") == ""


def test_6_exchange_success_and_cache(monkeypatch):
    _clear_caches()
    calls = []

    class FakeResponse:
        def __init__(self, payload): self._p = json.dumps(payload).encode()
        def read(self): return self._p
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        assert request.full_url == ca._TOKEN_EXCHANGE_URL
        assert request.headers["Authorization"] == "token ghu_raw"
        assert request.headers["Editor-version"] == ca._EDITOR_VERSION
        return FakeResponse({"token": "tid=x;proxy-ep=proxy.individual.githubcopilot.com",
                             "expires_at": time.time() + 3600})

    monkeypatch.setattr(ca.urllib.request, "urlopen", fake_urlopen)
    token, expires_at, base_url = ca.exchange_copilot_token("ghu_raw")
    assert token.startswith("tid=x")
    assert base_url == "https://api.individual.githubcopilot.com"
    assert expires_at > time.time()
    # Second call is served from cache — no second network round-trip.
    ca.exchange_copilot_token("ghu_raw")
    assert len(calls) == 1
    # endpoints.api wins over the proxy-ep derivation when present.
    _clear_caches()
    monkeypatch.setattr(ca.urllib.request, "urlopen", lambda r, timeout=None: FakeResponse(
        {"token": "tid=y;proxy-ep=proxy.ignored.com", "expires_at": time.time() + 3600,
         "endpoints": {"api": "https://api.corp.example.com/"}}))
    assert ca.exchange_copilot_token("ghu_other")[2] == "https://api.corp.example.com"


# A rejected token must not re-hit the network on every turn
def test_7_negative_cache(monkeypatch):
    _clear_caches()
    import urllib.error
    calls = []

    def boom(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(ca._TOKEN_EXCHANGE_URL, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(ca.urllib.request, "urlopen", boom)
    for _ in range(3):
        try:
            ca.exchange_copilot_token("ghu_bad")
        except ca.CopilotAuthError as exc:
            assert "401" in str(exc) or "not retrying" in str(exc)
    assert len(calls) == 1, "a 401 must be negatively cached, not retried every call"


# A failed exchange degrades to the raw token rather than dying
def test_8_credentials_fallback(monkeypatch):
    _clear_caches()
    monkeypatch.setattr(ca, "resolve_github_token", lambda: ("ghu_raw", "env"))
    monkeypatch.setattr(ca, "exchange_copilot_token",
                        lambda t, **kw: (_ for _ in ()).throw(ca.CopilotAuthError("offline")))
    assert ca.get_copilot_credentials() == ("ghu_raw", ca.COPILOT_DEFAULT_BASE_URL)
    monkeypatch.setattr(ca, "resolve_github_token", lambda: ("", ""))
    try:
        ca.get_copilot_credentials()
        assert False, "should raise when there is no token at all"
    except ca.CopilotAuthError as exc:
        assert "gg-agent login" in str(exc)


def test_9_request_headers():
    headers = ca.copilot_request_headers()
    # Without Copilot-Integration-Id the API rejects the request outright.
    assert headers["Copilot-Integration-Id"] == "vscode-chat"
    assert headers["Editor-Version"].startswith("vscode/")
    assert headers["x-initiator"] == "agent"
    assert ca.copilot_request_headers(is_agent_turn=False)["x-initiator"] == "user"


def test_10_profile_wiring(monkeypatch):
    _clear_caches()
    profile = get_provider_profile("copilot")
    assert profile is not None
    assert get_provider_profile("github-copilot") is profile, "alias should resolve"
    assert profile.api_mode == "chat_completions", "Copilot reuses the OpenAI transport"
    assert profile.default_headers["Copilot-Integration-Id"] == "vscode-chat"
    monkeypatch.setattr(ca, "get_copilot_credentials", lambda: ("live-token", "https://api.corp.example.com"))
    import gg_agent.providers as P
    monkeypatch.setattr(P, "get_copilot_credentials", lambda: ("live-token", "https://api.corp.example.com"))
    assert profile.resolve_credentials() == ("live-token", "https://api.corp.example.com")


# A bare GITHUB_TOKEN (usually exported for `gh`/CI) must not hijack auto-detection
def test_11_auto_detect_is_copilot_specific(monkeypatch, tmp_path):
    profile = get_provider_profile("copilot")
    monkeypatch.setattr(ca, "_CRED_FILES", (str(tmp_path / "absent.json"),))
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghu_generic")
    assert not profile.has_credentials()
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_explicit")
    assert profile.has_credentials()


# The Agent swaps its client when the profile hands back a new token
def test_12_agent_refreshes_expired_credentials():
    from gg_agent import Agent, ProviderProfile, register_provider, register_transport
    from gg_agent.transports.base import ProviderTransport
    from gg_agent.transports.types import NormalizedResponse

    built = []

    class RotatingProfile(ProviderProfile):
        counter = 0
        def resolve_credentials(self):
            RotatingProfile.counter += 1
            return f"token-{RotatingProfile.counter}", "https://host.example.com"

    class T(ProviderTransport):
        api_mode = property(lambda self: "rotating")
        def build_client(self, *, api_key, base_url, profile):
            built.append(api_key)
            return object()
        def convert_messages(self, m, **k): return m
        def convert_tools(self, t): return t
        def build_kwargs(self, model, messages, tools=None, **p): return {"model": model, "messages": messages}
        def call(self, client, **kw): return NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")
        def normalize_response(self, r): return r

    register_transport(T())
    register_provider(RotatingProfile(name="rotating", api_mode="rotating", default_model="r"))
    agent = Agent(provider="rotating")
    assert built == ["token-1"]
    agent.run("hi")
    assert built == ["token-1", "token-2"], "a changed credential must rebuild the client"
    assert agent.api_key == "token-2"
