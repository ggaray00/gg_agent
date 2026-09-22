"""Nous Portal provider profile."""

from typing import Any

from .. import register_provider
from ..base import ProviderProfile


class NousProfile(ProviderProfile):
    def build_extra_body(self, *, session_id: str | None = None, **ctx: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"tags": ["product=gg-agent"]}
        # Top-level session_id = sticky routing key, so cache breakpoints stay warm on
        # one upstream instance. The Portal rejects caller routing prefs (provider.*).
        if session_id:
            body["session_id"] = session_id
        return body

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                supports_reasoning: bool = False, **ctx: Any):
        if not supports_reasoning:
            return {}, {}
        if reasoning_config is None:
            return {"reasoning": {"enabled": True, "effort": "medium"}}, {}
        rc = dict(reasoning_config)
        # Some routes mandate reasoning and 400 on a disable; without the Portal's
        # capability catalog, omitting beats a 400 (the model just thinks).
        if rc.get("enabled") is False:
            return {}, {}
        return {"reasoning": rc}, {}


register_provider(NousProfile(
    name="nous", aliases=("nous-portal", "nousresearch"), display_name="Nous Research",
    description="Nous Research — Hermes model family", signup_url="https://nousresearch.com/",
    env_vars=("NOUS_API_KEY",), base_url="https://inference-api.nousresearch.com/v1",
    fallback_models=("hermes-3-405b", "hermes-3-70b"),
))
