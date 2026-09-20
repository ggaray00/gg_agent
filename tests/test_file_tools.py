"""File-tool path resolution: relative paths follow the agent's working dir."""
import asyncio
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gg_agent.tools.file_tools import _resolve, list_dir, read_file, write_file
from gg_agent.tools.registry import discover_builtin_tools, registry


def agent_at(path) -> types.SimpleNamespace:
    """The handlers only ever read ``.cwd``, so a stand-in beats building an Agent."""
    return types.SimpleNamespace(cwd=str(path))


def test_relative_path_anchors_on_agent_cwd(tmp_path):
    (tmp_path / "note.txt").write_text("hello")
    assert read_file("note.txt", parent_agent=agent_at(tmp_path)) == "1\thello"


def test_relative_path_without_agent_uses_process_cwd(tmp_path, monkeypatch):
    (tmp_path / "note.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)
    assert read_file("note.txt") == "1\thello"


def test_absolute_path_ignores_agent_cwd(tmp_path):
    target = tmp_path / "abs.txt"
    target.write_text("absolute")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert read_file(str(target), parent_agent=agent_at(elsewhere)) == "1\tabsolute"


def test_home_relative_path_ignores_agent_cwd(tmp_path):
    assert _resolve("~/x", agent_at(tmp_path)) == str(Path.home() / "x")


def test_write_creates_file_under_agent_cwd(tmp_path):
    out = write_file("sub/dir/new.txt", "body", parent_agent=agent_at(tmp_path))
    assert (tmp_path / "sub" / "dir" / "new.txt").read_text() == "body"
    assert str(tmp_path) in out


def test_list_dir_defaults_to_agent_cwd(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "child").mkdir()
    listing = list_dir(parent_agent=agent_at(tmp_path))
    assert listing.splitlines() == [f"{tmp_path}:", "a.txt", "child/"]


def test_dispatch_passes_the_agent_through(tmp_path):
    """The registry must hand parent_agent to the file tools (needs_agent=True)."""
    discover_builtin_tools()
    (tmp_path / "note.txt").write_text("dispatched")
    result = asyncio.run(
        registry.dispatch("read_file", {"path": "note.txt"}, agent=agent_at(tmp_path)))
    assert result == "1\tdispatched"


def test_missing_relative_file_reports_the_resolved_path(tmp_path):
    error = json.loads(read_file("nope.txt", parent_agent=agent_at(tmp_path)))
    assert error["error"] == f"not a file: {tmp_path / 'nope.txt'}"
