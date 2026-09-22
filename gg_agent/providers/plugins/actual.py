"""Actual Computer provider profile: hosted at api.actual.inc; local
(offline-mode client) inference opted into via ACTUAL_BASE_URL."""

from ...reasoning_effort import ACTUAL_RELAY_EFFORTS
from .. import register_provider
from ..base import ProviderProfile


def normalize_actual_base_url(url: str) -> str:
    """Bare hosts get ``/v1`` appended."""
    url = (url or "").strip().rstrip("/")
    return url if not url or url.endswith("/v1") else url + "/v1"


class ActualProfile(ProviderProfile):
    def resolve_base_url(self) -> str:
        return normalize_actual_base_url(super().resolve_base_url())

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        return ACTUAL_RELAY_EFFORTS


register_provider(ActualProfile(
    name="actual", aliases=("actual-computer", "actualcomputer", "aci"), display_name="Actual Computer",
    description="Actual Computer — hosted inference via api.actual.inc, or local via ACTUAL_BASE_URL",
    signup_url="https://actual.inc",
    env_vars=("ACTUAL_API_KEY", "ACTUAL_BASE_URL"), base_url="https://api.actual.inc/v1",
    api_mode="codex_responses",
))
