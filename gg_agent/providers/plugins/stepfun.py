"""StepFun provider profile."""

from .. import register_provider
from ..base import ProviderProfile

register_provider(ProviderProfile(
    name="stepfun", aliases=("step", "stepfun-coding-plan"), display_name="StepFun",
    env_vars=("STEPFUN_API_KEY",), base_url="https://api.stepfun.ai/step_plan/v1",
    default_aux_model="step-3.5-flash",
))
