"""FastAPI routes for the assistant."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    IndexResponse,
    ToolDescription,
)
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------- Helpers ----------

def _agent(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready.")
    return agent


def _mcp(request: Request):
    mgr = getattr(request.app.state, "mcp", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP manager not ready.")
    return mgr


def _indexer(request: Request):
    idx = getattr(request.app.state, "indexer", None)
    if idx is None:
        raise HTTPException(status_code=503, detail="Indexer not ready.")
    return idx


# ---------- Health ----------

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    s = get_settings()
    reachable = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{s.ollama_base_url}/api/tags")
            reachable = r.status_code == 200
    except Exception:
        reachable = False

    mgr = getattr(request.app.state, "mcp", None)
    tool_count = len(mgr.tools) if mgr else 0

    return HealthResponse(
        status="ok",
        ollama_reachable=reachable,
        ollama_model=s.ollama_model,
        workspace_root=str(s.workspace_root),
        tool_count=tool_count,
    )


# ---------- Tools ----------

@router.get("/tools", response_model=list[ToolDescription])
async def list_tools(request: Request) -> list[ToolDescription]:
    return [ToolDescription(**d) for d in _mcp(request).tool_descriptions()]


# ---------- Chat ----------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    agent = _agent(request)
    thread_id = req.ensure_thread_id()

    if req.stream:
        return StreamingResponse(
            _sse(agent.astream_events(req.message, thread_id), thread_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    final = await agent.ainvoke(req.message, thread_id)
    answer = ""
    tool_calls: list[dict[str, Any]] = []
    for m in final.get("messages", []):
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls.append({"name": tc.get("name"), "args": tc.get("args")})
        if m.__class__.__name__ == "AIMessage":
            answer = m.content
    return ChatResponse(thread_id=thread_id, answer=answer, tool_calls=tool_calls)


async def _sse(
    events: AsyncIterator[dict[str, Any]], thread_id: str
) -> AsyncIterator[bytes]:
    """Convert agent events into Server-Sent Events frames.

    Buffering rule: token chunks whose accumulated text *looks like* a
    tool-call JSON ({\"name\": ..., \"arguments\": ...}) are held back
    until either (a) the buffer no longer parses as tool-call JSON
    (false positive — flush) or (b) a tool_start event arrives
    (genuine tool call — discard the buffered tokens since the agent
    is about to dispatch the tool anyway).
    """
    yield f"event: meta\ndata: {json.dumps({'thread_id': thread_id})}\n\n".encode()

    token_buffer: list[str] = []

    def _looks_like_tool_call(text: str) -> bool:
        s = text.lstrip()
        if not s:
            return False
        if s.startswith("```"):
            return True
        if not s.startswith("{"):
            return False
        # Cheap heuristic: keep buffering as long as it could become a
        # ``{"name": ..., "arguments": ...}`` object.
        return '"name"' in s or len(s) < 24

    def _flush_tokens() -> list[bytes]:
        if not token_buffer:
            return []
        joined = "".join(token_buffer)
        token_buffer.clear()
        payload = json.dumps({"type": "token", "content": joined})
        return [f"event: token\ndata: {payload}\n\n".encode()]

    try:
        async for ev in events:
            kind = ev.get("type")

            if kind == "token":
                token_buffer.append(ev.get("content", ""))
                if _looks_like_tool_call("".join(token_buffer)):
                    continue
                for frame in _flush_tokens():
                    yield frame
                continue

            if kind == "tool_start":
                # A real tool dispatch — drop any buffered JSON tokens.
                token_buffer.clear()
            else:
                for frame in _flush_tokens():
                    yield frame

            payload = json.dumps(ev, default=str)
            yield f"event: {kind or 'message'}\ndata: {payload}\n\n".encode()

        for frame in _flush_tokens():
            yield frame
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("chat.stream_error", error=str(exc))
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n".encode()
    yield b"event: done\ndata: {}\n\n"


# ---------- History ----------

@router.get("/history/{thread_id}", response_model=HistoryResponse)
async def history(thread_id: str, request: Request) -> HistoryResponse:
    agent = _agent(request)
    msgs = await agent.get_history(thread_id)
    return HistoryResponse(thread_id=thread_id, messages=msgs)


# ---------- Indexing ----------

@router.post("/index", response_model=IndexResponse)
async def index_repo(request: Request) -> IndexResponse:
    idx = _indexer(request)
    result = idx.index_workspace()
    return IndexResponse(**result)


@router.get("/index/stats")
async def index_stats(request: Request) -> dict:
    return _indexer(request).stats()
