"""gg-agent — a bare-bones reconstruction of hermes-agent's core.

    from gg_agent import Agent
    with Agent(provider="anthropic") as agent:
        print(agent.ask("How many Python files are in this repo?"))
"""

from .agent import Agent, resolve_provider
from .loop import run_conversation
from .providers import ProviderProfile, get_provider_profile, list_providers, register_provider
from .tools.registry import registry, tool_error, tool_ok
from .transports import get_transport, register_transport

__version__ = "0.1.0"

__all__ = [
    "Agent", "resolve_provider", "run_conversation",
    "ProviderProfile", "get_provider_profile", "list_providers", "register_provider",
    "registry", "tool_error", "tool_ok",
    "get_transport", "register_transport",
]
