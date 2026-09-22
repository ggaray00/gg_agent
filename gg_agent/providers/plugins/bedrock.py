"""AWS Bedrock provider profile — the Converse API through boto3.

Auth is the standard AWS chain (env keys, ``AWS_PROFILE``, SSO, instance role),
or a Bedrock API key in ``AWS_BEARER_TOKEN_BEDROCK``. Region: AWS_REGION, then
AWS_DEFAULT_REGION, then the profile's configured region, then us-east-1.
Needs ``pip install boto3``.
"""

from __future__ import annotations

import os
from typing import Any

from .. import register_provider
from ..base import ProviderProfile


def resolve_bedrock_region() -> str:
    explicit = os.getenv("AWS_REGION", "").strip() or os.getenv("AWS_DEFAULT_REGION", "").strip()
    if explicit:
        return explicit
    try:
        import botocore.session
        return botocore.session.get_session().get_config_variable("region") or "us-east-1"
    except Exception:
        return "us-east-1"


class BedrockProfile(ProviderProfile):
    def resolve_credentials(self) -> tuple[str, str]:
        # boto3 resolves the real credentials itself; the "key" is only the optional
        # Bedrock API key, and the base URL carries the region.
        region = resolve_bedrock_region()
        return os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip(), f"https://bedrock-runtime.{region}.amazonaws.com"

    def has_credentials(self) -> bool:
        return any(os.getenv(v, "").strip() for v in
                   ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_WEB_IDENTITY_TOKEN_FILE"))

    def credential_status(self) -> str:
        try:
            import boto3
        except ImportError:
            return "boto3 not installed (pip install boto3)"
        if os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip():
            return f"Bedrock API key via AWS_BEARER_TOKEN_BEDROCK @ {resolve_bedrock_region()}"
        creds = boto3.Session().get_credentials()
        if creds is None:
            return "no AWS credentials found"
        return f"AWS credentials via {getattr(creds, 'method', 'default chain')} @ {resolve_bedrock_region()}"

    def fetch_models(self, **kwargs: Any):
        try:
            import boto3
            client = boto3.client("bedrock", region_name=resolve_bedrock_region())
            profiles = client.list_inference_profiles().get("inferenceProfileSummaries", [])
            return [p["inferenceProfileId"] for p in profiles if p.get("inferenceProfileId")]
        except Exception:
            return None


register_provider(BedrockProfile(
    name="bedrock", aliases=("aws", "aws-bedrock", "amazon-bedrock", "amazon"), display_name="AWS Bedrock",
    api_mode="bedrock_converse", env_vars=(), auth_type="aws_sdk",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    default_model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    default_max_tokens=8192,
    supports_vision=True,
))
