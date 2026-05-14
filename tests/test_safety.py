"""Tests for the path-sandboxing helpers used by the MCP servers."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcp_servers import _safety


def test_safe_resolve_inside_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.txt").write_text("hi")

    out = _safety.safe_resolve("a/b.txt")
    assert out == (tmp_path / "a" / "b.txt").resolve()


def test_safe_resolve_blocks_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        _safety.safe_resolve("../../etc/passwd")


def test_should_skip_handles_default_ignores():
    p = Path(".git/config")
    assert _safety.should_skip(p)
    assert not _safety.should_skip(Path("app/main.py"))
