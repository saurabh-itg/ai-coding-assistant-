"""Git MCP server.

Tools exposed:
    * recent_commits  — last N commits with author/date/message
    * changed_files   — files modified between two refs
    * branch_diff     — unified diff between two refs
    * blame_file      — git blame for a file
    * git_status      — porcelain status
    * current_branch  — name of HEAD
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_servers._safety import get_workspace_root, safe_resolve  # noqa: E402

mcp = FastMCP("git")


def _repo():
    """Lazily import GitPython and open the workspace repo."""
    import git  # type: ignore

    try:
        return git.Repo(get_workspace_root(), search_parent_directories=False)
    except git.InvalidGitRepositoryError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            f"Workspace '{get_workspace_root()}' is not a git repository."
        ) from exc


@mcp.tool()
def recent_commits(limit: int = 20, branch: Optional[str] = None) -> dict:
    """List the most recent commits.

    Args:
        limit: Maximum number of commits to return (default 20).
        branch: Branch / ref name. Defaults to current HEAD.
    """
    try:
        repo = _repo()
    except RuntimeError as exc:
        return {"error": str(exc)}

    rev = branch or repo.head.ref.name
    out: List[dict] = []
    for c in repo.iter_commits(rev, max_count=limit):
        out.append(
            {
                "sha": c.hexsha[:12],
                "author": f"{c.author.name} <{c.author.email}>",
                "date": c.committed_datetime.isoformat(),
                "message": c.message.strip().splitlines()[0] if c.message else "",
            }
        )
    return {"branch": rev, "commits": out}


@mcp.tool()
def changed_files(base: str = "HEAD~1", head: str = "HEAD") -> dict:
    """List files changed between ``base`` and ``head`` refs."""
    try:
        repo = _repo()
    except RuntimeError as exc:
        return {"error": str(exc)}

    try:
        diff = repo.git.diff("--name-status", f"{base}..{head}")
    except Exception as exc:
        return {"error": f"git diff failed: {exc}"}

    files: list[dict] = []
    for line in diff.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append({"status": parts[0], "path": parts[-1]})
    return {"base": base, "head": head, "files": files}


@mcp.tool()
def branch_diff(base: str = "HEAD~1", head: str = "HEAD", max_lines: int = 1500) -> dict:
    """Unified diff between two refs (truncated to ``max_lines``)."""
    try:
        repo = _repo()
    except RuntimeError as exc:
        return {"error": str(exc)}

    try:
        diff_text = repo.git.diff(f"{base}..{head}", unified=3)
    except Exception as exc:
        return {"error": f"git diff failed: {exc}"}

    lines = diff_text.splitlines()
    truncated = len(lines) > max_lines
    return {
        "base": base,
        "head": head,
        "truncated": truncated,
        "diff": "\n".join(lines[:max_lines]),
    }


@mcp.tool()
def blame_file(path: str, max_lines: int = 500) -> dict:
    """Return git blame for ``path`` (capped to ``max_lines``)."""
    target = safe_resolve(path)
    try:
        repo = _repo()
    except RuntimeError as exc:
        return {"error": str(exc)}

    rel = str(target.relative_to(get_workspace_root()))
    try:
        blame_raw = repo.git.blame("--line-porcelain", rel)
    except Exception as exc:
        return {"error": f"git blame failed: {exc}"}

    out: list[dict] = []
    current: dict = {}
    for ln in blame_raw.splitlines():
        if ln.startswith("\t"):
            current["text"] = ln[1:]
            out.append(current)
            current = {}
            if len(out) >= max_lines:
                break
            continue
        if not current:
            sha, *_ = ln.split()
            current["sha"] = sha[:12]
            continue
        if ln.startswith("author "):
            current["author"] = ln[len("author ") :]
        elif ln.startswith("summary "):
            current["summary"] = ln[len("summary ") :]

    return {"path": rel, "lines": out, "truncated": len(out) >= max_lines}


@mcp.tool()
def git_status() -> dict:
    """Return porcelain status of the working tree."""
    try:
        repo = _repo()
    except RuntimeError as exc:
        return {"error": str(exc)}

    try:
        raw = repo.git.status("--porcelain=v1", "--branch")
    except Exception as exc:
        return {"error": f"git status failed: {exc}"}
    return {"status": raw}


@mcp.tool()
def current_branch() -> dict:
    """Return the current branch name (or detached SHA)."""
    try:
        repo = _repo()
    except RuntimeError as exc:
        return {"error": str(exc)}

    try:
        name = repo.active_branch.name
    except TypeError:
        name = repo.head.commit.hexsha[:12]
    return {"branch": name}


if __name__ == "__main__":
    mcp.run()
