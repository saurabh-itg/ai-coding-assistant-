"""Filesystem MCP server.

Tools exposed:
    * list_files     — list entries beneath a path (workspace-sandboxed)
    * read_file      — read a UTF-8 text file (size-limited)
    * search_code    — ripgrep-style regex search across the workspace
    * file_stats     — basic metadata about a path
    * list_directory_tree — pretty-printed recursive tree

All paths are sandboxed inside ``WORKSPACE_ROOT``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

# Make the project importable when this file is launched as a subprocess.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_servers._safety import (  # noqa: E402
    DEFAULT_IGNORES,
    get_workspace_root,
    safe_resolve,
    should_skip,
)

mcp = FastMCP("filesystem")

MAX_READ_BYTES = 256 * 1024  # 256 KB per read
MAX_SEARCH_HITS = 200


@mcp.tool()
def list_files(path: str = ".", include_hidden: bool = False) -> dict:
    """List files and directories directly beneath ``path``.

    Args:
        path: Relative or absolute path inside the workspace. Defaults to root.
        include_hidden: If False, dotfiles/dirs and well-known build dirs are
            hidden.

    Returns a dict with ``path``, ``files`` (list of names), and ``dirs``.
    """
    target = safe_resolve(path)
    if not target.exists():
        return {"error": f"Path does not exist: {target}"}
    if not target.is_dir():
        return {"error": f"Not a directory: {target}"}

    files: List[str] = []
    dirs: List[str] = []
    for entry in sorted(target.iterdir()):
        if not include_hidden and entry.name.startswith("."):
            continue
        if not include_hidden and entry.name in DEFAULT_IGNORES:
            continue
        (dirs if entry.is_dir() else files).append(entry.name)

    return {
        "path": str(target.relative_to(get_workspace_root())) or ".",
        "dirs": dirs,
        "files": files,
    }


@mcp.tool()
def read_file(path: str, max_bytes: int = MAX_READ_BYTES) -> dict:
    """Read a UTF-8 text file. Returns up to ``max_bytes`` bytes.

    Args:
        path: Relative or absolute file path inside the workspace.
        max_bytes: Hard cap on bytes returned. Default 256 KB.
    """
    target = safe_resolve(path)
    if not target.exists():
        return {"error": f"File does not exist: {target}"}
    if not target.is_file():
        return {"error": f"Not a file: {target}"}

    cap = min(max_bytes, MAX_READ_BYTES)
    try:
        data = target.read_bytes()[:cap]
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        return {"error": f"Read failed: {exc}"}

    return {
        "path": str(target.relative_to(get_workspace_root())),
        "size": target.stat().st_size,
        "truncated": target.stat().st_size > cap,
        "content": text,
    }


@mcp.tool()
def search_code(
    pattern: str,
    path: str = ".",
    glob: str = "",
    case_sensitive: bool = False,
    max_results: int = MAX_SEARCH_HITS,
) -> dict:
    """Regex search across files in the workspace (pure Python, no rg required).

    Args:
        pattern: Python regex pattern.
        path: Subdirectory or file to scope the search to.
        glob: Optional filename suffix filter, e.g. ``.py`` or ``.md``.
        case_sensitive: Match case if True.
        max_results: Stop after this many matches.
    """
    base = safe_resolve(path)
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return {"error": f"Bad regex: {exc}"}

    iterable = [base] if base.is_file() else base.rglob("*")

    hits: list[dict] = []
    for file in iterable:
        if len(hits) >= max_results:
            break
        if not file.is_file():
            continue
        if should_skip(file.relative_to(get_workspace_root())):
            continue
        if glob and not file.name.endswith(glob):
            continue
        try:
            with file.open("r", encoding="utf-8", errors="ignore") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if regex.search(line):
                        hits.append(
                            {
                                "file": str(file.relative_to(get_workspace_root())),
                                "line": lineno,
                                "text": line.rstrip()[:400],
                            }
                        )
                        if len(hits) >= max_results:
                            break
        except Exception:
            continue

    return {
        "pattern": pattern,
        "path": str(base.relative_to(get_workspace_root())) or ".",
        "matches": hits,
        "truncated": len(hits) >= max_results,
    }


@mcp.tool()
def file_stats(path: str) -> dict:
    """Return size, mtime, and type for a path."""
    target = safe_resolve(path)
    if not target.exists():
        return {"error": f"Path does not exist: {target}"}
    st = target.stat()
    return {
        "path": str(target.relative_to(get_workspace_root())),
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "size": st.st_size,
        "modified_unix": int(st.st_mtime),
    }


@mcp.tool()
def list_directory_tree(path: str = ".", max_depth: int = 3) -> dict:
    """Render a pretty recursive tree, capped at ``max_depth``."""
    base = safe_resolve(path)
    if not base.is_dir():
        return {"error": f"Not a directory: {base}"}

    lines: list[str] = []

    def walk(p: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                [e for e in p.iterdir() if e.name not in DEFAULT_IGNORES and not e.name.startswith(".")]
            )
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(entry, depth + 1, prefix + extension)

    rel = str(base.relative_to(get_workspace_root())) or "."
    lines.append(f"{rel}/")
    walk(base, 1, "")
    return {"path": rel, "tree": "\n".join(lines)}


if __name__ == "__main__":
    mcp.run()
