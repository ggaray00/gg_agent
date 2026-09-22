"""Ollama (local) provider profile.

A convenience over ``custom``: the default local endpoint, no configuration.
Anything else OpenAI-compatible (vLLM, llama.cpp, LM Studio) goes through ``custom``.
"""

from .. import register_provider
from .custom import CustomProfile

register_provider(CustomProfile(
    name="ollama",
    display_name="Ollama (local)",
    description="Local Ollama server on :11434",
    env_vars=("OLLAMA_BASE_URL",),           # no key needed
    base_url="http://localhost:11434/v1",
    default_model="qwen2.5-coder:7b",
    fixed_temperature=0.0,
    # Without a max_tokens Ollama falls back to num_predict=128 and truncates.
    default_max_tokens=65536,
))
