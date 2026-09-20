"""Read / write / list files.

Mirrors hermes-agent: tools/file_tools.py, tools/file_operations_*.py
(which add read-tracking, edit approval, checkpoints and path-security).
"""

from __future__ import annotations

import os
from typing import Any

from .registry import registry, tool_error

MAX_READ_CHARS = 60_000


def _resolve(path: str, parent_agent: Any = None) -> str:
    """Absolute path for a tool argument, anchored on the agent's working dir.

    A relative path means "relative to the directory the agent was told it is
    working in" — the same one in its system prompt and the shell tool's default
    — which is only the process cwd when nobody passed ``Agent(cwd=...)``.
    Absolute and ``~`` paths are unaffected.
    """
    expanded = os.path.expanduser(path)
    base = getattr(parent_agent, "cwd", None)
    if os.path.isabs(expanded) or not base:
        return os.path.abspath(expanded)
    return os.path.normpath(os.path.join(base, expanded))


def read_file(path: str, offset: int = 0, limit: int = 2000, parent_agent: Any = None) -> str:
    full = _resolve(path, parent_agent)
    if not os.path.isfile(full):
        return tool_error(f"not a file: {full}")
    try:
        with open(full, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return tool_error(f"could not read {full}: {exc}")

    start = max(int(offset or 0), 0)
    chunk = lines[start:start + max(int(limit or 2000), 1)]
    # Line numbers let the model quote precise locations back to the user.
    body = "".join(f"{start + i + 1}\t{line}" for i, line in enumerate(chunk))
    if len(body) > MAX_READ_CHARS:
        body = body[:MAX_READ_CHARS] + "\n… [truncated — re-read with offset/limit]"
    return body or "(empty file)"


def write_file(path: str, content: str, parent_agent: Any = None) -> str:
    full = _resolve(path, parent_agent)
    try:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content or "")
    except OSError as exc:
        return tool_error(f"could not write {full}: {exc}")
    return f"wrote {len(content or '')} chars to {full}"


def list_dir(path: str = ".", parent_agent: Any = None) -> str:
    full = _resolve(path, parent_agent)
    if not os.path.isdir(full):
        return tool_error(f"not a directory: {full}")
    entries = []
    for name in sorted(os.listdir(full)):
        child = os.path.join(full, name)
        entries.append(f"{name}/" if os.path.isdir(child) else name)
    return f"{full}:\n" + "\n".join(entries[:500])


registry.register_toolset("files", "Read, write and list files")

registry.register(
    name="read_file", toolset="files", emoji="📄", handler=read_file,
    needs_agent=True,          # relative paths resolve against parent_agent.cwd
    schema={
        "name": "read_file",
        "description": "Read a text file with line numbers. Use offset/limit for large files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "File path: absolute, ~-relative, or relative to the "
                                        "working directory."},
                "offset": {"type": "integer", "description": "0-based first line to read."},
                "limit": {"type": "integer", "description": "Maximum lines to read (default 2000)."},
            },
            "required": ["path"],
        },
    },
)

registry.register(
    name="write_file", toolset="files", emoji="✍️", handler=write_file,
    needs_agent=True,
    schema={
        "name": "write_file",
        "description": "Write (overwrite) a text file, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "File path: absolute, ~-relative, or relative to the "
                                        "working directory."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
)

registry.register(
    name="list_dir", toolset="files", emoji="📁", handler=list_dir,
    needs_agent=True,
    schema={
        "name": "list_dir",
        "description": "List the entries of a directory (directories end with '/').",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path; defaults to the working directory."}},
            "required": [],
        },
    },
)
