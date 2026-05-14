"""Shared safety helpers for MCP servers (path sandboxing, etc.)."""
from __future__ import annotations

import os
from pathlib import Path


def get_workspace_root() -> Path:
    """Workspace root, taken from WORKSPACE_ROOT env var.

    Each MCP server runs as a subprocess that inherits the parent's
    environment, so the same env var seen by the FastAPI app is visible
    here too.
    """
    root = os.environ.get("WORKSPACE_ROOT")
    if not root:
        return Path.cwd().resolve()
    return Path(root).expanduser().resolve()


def safe_resolve(relative_or_absolute: str) -> Path:
    """Resolve a user-provided path and ensure it stays inside the workspace.

    Raises ``ValueError`` on path-traversal attempts or paths that fall
    outside the workspace root.
    """
    root = get_workspace_root()
    raw = Path(relative_or_absolute)
    candidate = (root / raw if not raw.is_absolute() else raw).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover — defensive
        raise ValueError(
            f"Path '{relative_or_absolute}' escapes the workspace root "
            f"('{root}'). Access denied."
        ) from exc

    return candidate


# Files / dirs that are nearly always uninteresting to scan.
DEFAULT_IGNORES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".idea",
    ".vscode",
    "data",
    "chroma",
}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & DEFAULT_IGNORES)
