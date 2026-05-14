"""Tests for terminal MCP server safety logic."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcp_servers import terminal_server as ts


def _unwrap(tool):
    return getattr(tool, "fn", tool)


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    # ``python`` is portable across Windows and Unix; ``echo`` is a shell
    # builtin on Windows so we can't rely on it with shell=False.
    monkeypatch.setenv("TERMINAL_ALLOWLIST", "python")
    monkeypatch.setenv("TERMINAL_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("TERMINAL_MAX_OUTPUT_BYTES", "10000")
    return tmp_path


def test_blocks_disallowed(workspace):
    out = _unwrap(ts.run_command)("rm -rf /")
    assert "error" in out
    assert "allow-list" in out["error"]


def test_runs_allowed_python(workspace):
    # ``python -V`` is portable and avoids quoting differences between
    # POSIX and Windows shells.
    out = _unwrap(ts.run_command)("python -V")
    assert out.get("exit_code") == 0, out
    combined = (out.get("stdout") or "") + (out.get("stderr") or "")
    assert "Python" in combined


def test_blocks_cwd_escape(workspace):
    out = _unwrap(ts.run_command)("python --version", cwd="../..")
    assert "error" in out
    assert "escapes" in out["error"]


def test_list_allowed_returns_config(workspace):
    out = _unwrap(ts.list_allowed)()
    assert "python" in out["allowed"]
    assert out["timeout_seconds"] == 10
