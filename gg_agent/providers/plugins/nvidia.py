"""NVIDIA NIM provider profile."""

from typing import Any

from .. import register_provider
from ..base import ProviderProfile


class NvidiaProviderProfile(ProviderProfile):
    """NIM validates tool messages strictly: ``name`` on a role=tool turn is a 400."""

    @staticmethod
    def _needs_strip(msg: Any) -> bool:
        return isinstance(msg, dict) and msg.get("role") == "tool" and ("name" in msg or "tool_name" in msg)

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Copy-on-write: only the tool turns that lose a field are copied.
        if not any(self._needs_strip(m) for m in messages):
            return messages
        return [{k: v for k, v in m.items() if k not in ("name", "tool_name")} if self._needs_strip(m) else m
                for m in messages]


register_provider(NvidiaProviderProfile(
    name="nvidia", aliases=("nvidia-nim",), display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference", signup_url="https://build.nvidia.com/",
    env_vars=("NVIDIA_API_KEY",), base_url="https://integrate.api.nvidia.com/v1",
    fallback_models=("nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.3-70b-instruct"),
    default_max_tokens=16384,
))
