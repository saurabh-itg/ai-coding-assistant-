"""Terminal MCP server.

Tools exposed:
    * run_command — execute a shell command, *but only* if its first token
                    is in ``TERMINAL_ALLOWLIST`` (env var).
    * list_allowed — return the current allow-list so the agent knows what
                     it can run.

Other safety properties:
    * Commands are executed with ``shell=False`` (no metacharacters).
    * Working directory is forced to ``WORKSPACE_ROOT``.
    * Output is captured with a hard timeout and a hard byte cap.
    * stdin is closed (no interactive prompts).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_servers._safety import get_workspace_root  # noqa: E402

mcp = FastMCP("terminal")


def _allowlist() -> set[str]:
    raw = os.environ.get(
        "TERMINAL_ALLOWLIST",
        "git,python,pytest,pip,ls,dir,cat,type,grep,rg,echo,pwd,cd,docker,kubectl,npm,node",
    )
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def _timeout() -> int:
    try:
        return int(os.environ.get("TERMINAL_TIMEOUT_SECONDS", "60"))
    except ValueError:
        return 60


def _max_bytes() -> int:
    try:
        return int(os.environ.get("TERMINAL_MAX_OUTPUT_BYTES", "200000"))
    except ValueError:
        return 200_000


def _split(command: str) -> List[str]:
    """Parse a command string into argv. ``posix=False`` keeps Windows paths intact."""
    return shlex.split(command, posix=os.name != "nt")


@mcp.tool()
def run_command(command: str, cwd: str = ".") -> dict:
    """Execute an allow-listed shell command and capture its output.

    Args:
        command: Full command line (e.g. ``"pytest -k auth -q"``).
        cwd: Working directory, relative to the workspace root.

    The first token of ``command`` is checked against the allow-list.
    Returns ``stdout``, ``stderr``, ``exit_code``, and a ``truncated`` flag.
    """
    if not command.strip():
        return {"error": "Empty command."}

    try:
        argv = _split(command)
    except ValueError as exc:
        return {"error": f"Could not parse command: {exc}"}

    head = Path(argv[0]).name.lower()
    head_no_ext = head.rsplit(".", 1)[0]
    allow = _allowlist()
    if head_no_ext not in allow and head not in allow:
        return {
            "error": (
                f"Executable '{argv[0]}' is not on the allow-list. "
                f"Allowed: {sorted(allow)}"
            )
        }

    workdir = (get_workspace_root() / cwd).resolve()
    try:
        workdir.relative_to(get_workspace_root())
    except ValueError:
        return {"error": f"cwd '{cwd}' escapes the workspace."}
    if not workdir.exists():
        return {"error": f"cwd does not exist: {workdir}"}

    cap = _max_bytes()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(workdir),
            capture_output=True,
            timeout=_timeout(),
            stdin=subprocess.DEVNULL,
            text=False,
            shell=False,
        )
    except FileNotFoundError:
        return {"error": f"Executable not found on PATH: {argv[0]}"}
    except subprocess.TimeoutExpired as exc:
        return {
            "error": f"Command timed out after {_timeout()}s",
            "stdout": (exc.stdout or b"")[:cap].decode("utf-8", errors="replace"),
            "stderr": (exc.stderr or b"")[:cap].decode("utf-8", errors="replace"),
        }

    stdout = proc.stdout[:cap].decode("utf-8", errors="replace")
    stderr = proc.stderr[:cap].decode("utf-8", errors="replace")
    truncated = len(proc.stdout) > cap or len(proc.stderr) > cap
    return {
        "command": command,
        "cwd": str(workdir.relative_to(get_workspace_root())) or ".",
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
    }


@mcp.tool()
def list_allowed() -> dict:
    """Return the current command allow-list and runtime limits."""
    return {
        "allowed": sorted(_allowlist()),
        "timeout_seconds": _timeout(),
        "max_output_bytes": _max_bytes(),
    }


if __name__ == "__main__":
    mcp.run()
