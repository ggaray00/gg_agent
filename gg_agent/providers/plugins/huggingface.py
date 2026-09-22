"""Hugging Face Inference provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="huggingface", aliases=("hf", "hugging-face", "huggingface-hub"),
    display_name="HuggingFace", description="HuggingFace Inference API",
    signup_url="https://huggingface.co/settings/tokens",
    env_vars=("HF_TOKEN",), base_url="https://router.huggingface.co/v1",
    fallback_models=("Qwen/Qwen3.5-72B-Instruct", "deepseek-ai/DeepSeek-V3.2"),
))
