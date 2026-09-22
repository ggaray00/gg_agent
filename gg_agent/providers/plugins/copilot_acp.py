"""GitHub Copilot ACP provider profile.

Not OpenAI-over-HTTP: it drives the Copilot CLI (``copilot --acp --stdio``) as a
subprocess speaking the Agent Client Protocol, so the profile supplies its own
client through ``create_client``. The CLI brings its own model and subscription;
nothing to configure beyond having it installed and signed in.
"""

from __future__ import annotations

import os
import shlex
import shutil
from typing import Any

from .. import register_provider
from ..base import ProviderProfile


class CopilotACPProfile(ProviderProfile):
    def command(self) -> tuple[str, list[str]]:
        cmd = next((os.getenv(v, "").strip() for v in self.process_command_env_vars if os.getenv(v, "").strip()),
                   self.process_command)
        args = shlex.split(os.getenv(self.process_args_env_var, "").strip()) or list(self.process_args)
        return cmd, args

    def create_client(self, **client_kwargs: Any) -> Any:
        from ...home import get_working_dir
        from ..copilot_acp_client import CopilotACPClient
        command, args = self.command()
        return CopilotACPClient(command=command, args=args, cwd=client_kwargs.get("cwd") or get_working_dir())

    def has_credentials(self) -> bool:
        return shutil.which(self.command()[0]) is not None

    def credential_status(self) -> str:
        command, args = self.command()
        path = shutil.which(command)
        return f"{path} {' '.join(args)}" if path else f"`{command}` not on PATH (npm install -g @github/copilot)"

    def fetch_models(self, **kwargs: Any):
        return None                          # the ACP session reports its own models


register_provider(CopilotACPProfile(
    name="copilot-acp", aliases=("github-copilot-acp", "copilot-acp-agent"), display_name="GitHub Copilot (ACP)",
    api_mode="chat_completions", env_vars=(), base_url="acp://copilot", auth_type="external_process",
    default_model="copilot-acp",             # = the CLI's session default
    process_command="copilot", process_args=("--acp", "--stdio"),
    process_command_env_vars=("GG_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
    process_args_env_var="GG_COPILOT_ACP_ARGS",
))
