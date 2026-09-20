"""Shell execution tool.

Mirrors hermes-agent: tools/terminal_tool.py (which additionally keeps a
persistent per-task_id session, approval callbacks and a sandbox).
"""

from __future__ import annotations

import asyncio
import os

from ..home import get_working_dir
from .registry import registry, tool_error

DEFAULT_TIMEOUT = 120
MAX_OUTPUT_CHARS = 20_000


async def run_shell(command: str, cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a shell command without blocking the event loop.

    Natively async rather than ``subprocess.run`` in a thread: a 10-minute build
    would otherwise pin a worker thread for its whole duration, and on timeout
    there would be no handle to kill the process with.
    """
    if not command or not command.strip():
        return tool_error("command is required")
    workdir = os.path.abspath(os.path.expanduser(cwd)) if cwd else get_working_dir()
    if not os.path.isdir(workdir):
        return tool_error(f"cwd does not exist: {workdir}")

    limit = min(int(timeout or DEFAULT_TIMEOUT), 600)
    try:
        proc = await asyncio.create_subprocess_shell(
            command, cwd=workdir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return tool_error(f"command failed to start: {exc}")

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=limit)
    except asyncio.TimeoutError:
        # Kill the process group's leader; a hung child must not outlive the turn.
        _terminate(proc)
        return tool_error(f"command timed out after {limit}s")
    except asyncio.CancelledError:
        _terminate(proc)
        raise

    stdout = (out or b"").decode("utf-8", errors="replace")
    stderr = (err or b"").decode("utf-8", errors="replace")
    body = stdout + (("\n[stderr]\n" + stderr) if stderr else "")
    if len(body) > MAX_OUTPUT_CHARS:
        body = body[:MAX_OUTPUT_CHARS] + "\n… [output truncated]"
    return f"exit_code={proc.returncode}\ncwd={workdir}\n\n{body or '(no output)'}"


def _terminate(proc) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass                     # already gone
    except Exception:
        pass


registry.register_toolset("shell", "Run shell commands")
registry.register(
    name="run_shell",
    toolset="shell",
    emoji="💻",
    schema={
        "name": "run_shell",
        "description": (
            "Run a shell command and return its exit code, stdout and stderr. "
            "Use for file inspection, git, builds, tests — anything a terminal can do. "
            "Long output is truncated, so prefer targeted commands (head, grep, sed -n)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "cwd": {"type": "string", "description": "Working directory. Defaults to the process cwd."},
                "timeout": {"type": "integer", "description": f"Seconds before the command is killed (default {DEFAULT_TIMEOUT}, max 600)."},
            },
            "required": ["command"],
        },
    },
    handler=run_shell,
)
