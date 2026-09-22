"""Alibaba Cloud DashScope provider profiles: pay-as-you-go (intl + CN), the Model
Studio Token Plan, and the Coding Plan tier. Names match models.dev catalog keys.

The CN / plan profiles check their own key first and keep the shared vars as
ordered fallbacks, so a user configured with the shared key keeps working.
"""

from .. import register_provider
from ..base import ProviderProfile

_SIGNUP = "https://help.aliyun.com/zh/model-studio/"

register_provider(ProviderProfile(
    name="alibaba", aliases=("dashscope", "alibaba-cloud", "qwen-dashscope"),
    display_name="Alibaba Cloud DashScope",
    env_vars=("DASHSCOPE_API_KEY",), base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
))
register_provider(ProviderProfile(
    name="alibaba-cn", aliases=("dashscope-cn", "alibaba-cloud-cn"),
    display_name="Alibaba Cloud DashScope (China)", description="Alibaba Cloud DashScope, mainland-China endpoint",
    env_vars=("DASHSCOPE_API_KEY", "DASHSCOPE_CN_BASE_URL"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
))
register_provider(ProviderProfile(
    name="alibaba-token-plan", aliases=("dashscope-token-plan",), display_name="Alibaba Cloud (Token Plan)",
    description="Alibaba Cloud Model Studio Token Plan (flat-token tier)", signup_url=_SIGNUP,
    env_vars=("ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_BASE_URL"),
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
))
register_provider(ProviderProfile(
    name="alibaba-token-plan-cn", aliases=("dashscope-token-plan-cn",),
    display_name="Alibaba Cloud (Token Plan, China)",
    description="Alibaba Cloud Model Studio Token Plan, mainland-China endpoint", signup_url=_SIGNUP,
    env_vars=("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_API_KEY", "ALIBABA_TOKEN_PLAN_CN_BASE_URL"),
    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
))
register_provider(ProviderProfile(
    name="alibaba-coding-plan", aliases=("alibaba_coding", "alibaba-coding", "dashscope-coding"),
    display_name="Alibaba Cloud (Coding Plan)", description="Alibaba Cloud Coding Plan (dedicated coding tier)",
    signup_url=_SIGNUP,
    env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_CODING_PLAN_BASE_URL"),
    base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
))
register_provider(ProviderProfile(
    name="alibaba-coding-plan-cn", aliases=("alibaba-coding-cn", "dashscope-coding-cn"),
    display_name="Alibaba Cloud (Coding Plan, China)", description="Alibaba Cloud Coding Plan, mainland-China endpoint",
    signup_url=_SIGNUP,
    env_vars=("ALIBABA_CODING_PLAN_CN_API_KEY", "ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY",
              "ALIBABA_CODING_PLAN_CN_BASE_URL"),
    base_url="https://coding.dashscope.aliyuncs.com/v1",
))
