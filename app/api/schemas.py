"""Pydantic request / response models for the HTTP API."""
from __future__ import annotations

import uuid
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message.")
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation id. Auto-generated if missing.",
    )
    stream: bool = Field(default=True, description="Stream tokens via SSE.")

    def ensure_thread_id(self) -> str:
        if self.thread_id and self.thread_id.strip():
            return self.thread_id.strip()
        return f"thread-{uuid.uuid4().hex[:12]}"


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    tool_calls: List[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    ollama_model: str
    workspace_root: str
    tool_count: int


class ToolDescription(BaseModel):
    name: str
    description: str
    args_schema: dict[str, Any] = {}


class IndexResponse(BaseModel):
    files_indexed: int
    chunks_indexed: int
    workspace: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: List[dict[str, Any]]
