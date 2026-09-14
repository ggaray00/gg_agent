"""Built-in tools. Importing a module here registers its tools as a side effect."""

from .registry import discover_builtin_tools, registry, tool_error, tool_ok  # noqa: F401
