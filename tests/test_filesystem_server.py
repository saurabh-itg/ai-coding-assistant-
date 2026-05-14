"""Tests for filesystem MCP tool implementations (called directly)."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcp_servers import filesystem_server as fs


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "src" / "b.py").write_text("import os\n# TODO fix\n")
    (tmp_path / "README.md").write_text("# project")
    return tmp_path


def _unwrap(tool):
    """FastMCP wraps tools — unwrap to the original Python function."""
    return getattr(tool, "fn", tool)


def test_list_files(workspace):
    out = _unwrap(fs.list_files)(".")
    assert "README.md" in out["files"]
    assert "src" in out["dirs"]


def test_read_file(workspace):
    out = _unwrap(fs.read_file)("src/a.py")
    assert "hello" in out["content"]
    assert out["truncated"] is False


def test_search_code_finds_todo(workspace):
    out = _unwrap(fs.search_code)("TODO", path=".", glob=".py")
    paths = {m["file"] for m in out["matches"]}
    assert any("b.py" in p for p in paths)


def test_search_code_blocks_bad_regex(workspace):
    out = _unwrap(fs.search_code)("(", path=".")
    assert "error" in out


def test_read_file_blocks_traversal(workspace):
    with pytest.raises(ValueError):
        _unwrap(fs.read_file)("../../etc/passwd")
