"""Manager that spawns MCP server subprocesses and exposes their tools.

Uses :class:`langchain_mcp_adapters.client.MultiServerMCPClient` which
handles the stdio transport and converts MCP tools into LangChain
``BaseTool`` instances ready to be bound to a LangGraph agent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_SERVER_DIR = Path(__file__).resolve().parents[2] / "mcp_servers"


def _server_config() -> Dict[str, Dict[str, Any]]:
    """Build the connection config that ``MultiServerMCPClient`` consumes."""
    settings = get_settings()
    python = sys.executable or "python"

    # All three servers inherit the parent process's environment, so they
    # see WORKSPACE_ROOT, TERMINAL_ALLOWLIST, etc. We make this explicit
    # so values still propagate when started via uvicorn workers.
    env = {
        **os.environ,
        "WORKSPACE_ROOT": str(settings.workspace_root),
        "TERMINAL_ALLOWLIST": settings.terminal_allowlist,
        "TERMINAL_TIMEOUT_SECONDS": str(settings.terminal_timeout_seconds),
        "TERMINAL_MAX_OUTPUT_BYTES": str(settings.terminal_max_output_bytes),
        "PYTHONPATH": str(_SERVER_DIR.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }

    return {
        "filesystem": {
            "command": python,
            "args": [str(_SERVER_DIR / "filesystem_server.py")],
            "transport": "stdio",
            "env": env,
        },
        "git": {
            "command": python,
            "args": [str(_SERVER_DIR / "git_server.py")],
            "transport": "stdio",
            "env": env,
        },
        "terminal": {
            "command": python,
            "args": [str(_SERVER_DIR / "terminal_server.py")],
            "transport": "stdio",
            "env": env,
        },
    }


class MCPManager:
    """Lifecycle wrapper around :class:`MultiServerMCPClient`."""

    def __init__(self) -> None:
        self._client: MultiServerMCPClient | None = None
        self._tools: List[BaseTool] = []

    async def start(self) -> None:
        if self._client is not None:
            return
        cfg = _server_config()
        logger.info("mcp.start", servers=list(cfg.keys()))
        self._client = MultiServerMCPClient(cfg)
        # ``get_tools`` opens an ephemeral session per server, lists tools,
        # and returns LangChain BaseTool wrappers that re-open a session
        # on each call. That's exactly what we want for a long-lived
        # FastAPI process — no shared, fragile session state.
        self._tools = await self._client.get_tools()
        logger.info("mcp.ready", tool_count=len(self._tools), tools=[t.name for t in self._tools])

    async def stop(self) -> None:
        # MultiServerMCPClient's tools open / close stdio per call, so
        # there is nothing persistent to tear down here. We keep the
        # method for symmetry / future transports (e.g. SSE, HTTP).
        self._client = None
        self._tools = []
        logger.info("mcp.stopped")

    @property
    def tools(self) -> List[BaseTool]:
        return list(self._tools)

    def tool_descriptions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in self._tools:
            schema: Dict[str, Any] = {}
            args = getattr(t, "args_schema", None)
            if args is None:
                pass
            elif isinstance(args, dict):
                schema = args
            elif hasattr(args, "model_json_schema"):
                schema = args.model_json_schema()
            elif hasattr(args, "schema"):  # pydantic v1 fallback
                try:
                    schema = args.schema()
                except Exception:
                    schema = {}
            out.append(
                {
                    "name": t.name,
                    "description": (t.description or "").strip(),
                    "args_schema": schema,
                }
            )
        return out
