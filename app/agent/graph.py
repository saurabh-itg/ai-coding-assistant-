"""LangGraph ReAct agent wired up to MCP tools and SQLite memory."""
from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent

from app.agent.llm import build_llm
from app.agent.prompts import SYSTEM_PROMPT
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class AssistantAgent:
    """Encapsulates the LangGraph agent, its tools and its persistent memory."""

    def __init__(self, tools: List[BaseTool]) -> None:
        self._tools = tools
        self._exit_stack = AsyncExitStack()
        self._graph = None
        self._checkpointer: AsyncSqliteSaver | None = None

    async def start(self) -> None:
        if self._graph is not None:
            return
        s = get_settings()
        s.ensure_dirs()

        # AsyncSqliteSaver.from_conn_string returns an async context manager
        # that yields the saver. We enter it through an exit-stack so the
        # underlying SQLite connection can be closed cleanly on shutdown.
        cm = AsyncSqliteSaver.from_conn_string(str(s.sqlite_path))
        self._checkpointer = await self._exit_stack.enter_async_context(cm)

        llm = build_llm()
        self._graph = create_react_agent(
            model=llm,
            tools=self._tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
        )
        logger.info("agent.ready", tool_count=len(self._tools))

    async def stop(self) -> None:
        await self._exit_stack.aclose()
        self._graph = None
        self._checkpointer = None
        logger.info("agent.stopped")

    @property
    def graph(self):
        if self._graph is None:
            raise RuntimeError("Agent not started. Call start() first.")
        return self._graph

    # ---------- Public API ----------

    async def ainvoke(self, message: str, thread_id: str) -> dict:
        """Run a single user turn and return the final state."""
        s = get_settings()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": s.agent_recursion_limit,
        }
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        return result

    async def astream_events(
        self, message: str, thread_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream rich events: token deltas, tool starts, tool ends.

        Yields plain dicts that the API layer can serialise as SSE.
        """
        s = get_settings()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": s.agent_recursion_limit,
        }
        async for event in self.graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is not None and getattr(chunk, "content", ""):
                    yield {"type": "token", "content": chunk.content}

            elif kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "name": event.get("name"),
                    "input": event["data"].get("input", {}),
                }

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                if isinstance(output, ToolMessage):
                    payload = output.content
                else:
                    payload = str(output)
                yield {
                    "type": "tool_end",
                    "name": event.get("name"),
                    "output": payload[:4000] if isinstance(payload, str) else payload,
                }

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                # Surface the assistant's final consolidated answer so the
                # client can replace any partial-token render with the
                # authoritative one.
                output = event["data"].get("output", {})
                msgs = output.get("messages", []) if isinstance(output, dict) else []
                if msgs and isinstance(msgs[-1], AIMessage):
                    yield {"type": "final", "content": msgs[-1].content}

    async def get_history(self, thread_id: str) -> list[dict]:
        """Return a serialisable view of the conversation for a thread."""
        if self._graph is None:
            return []
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = await self._graph.aget_state(config)
        except Exception:
            return []
        if state is None or not getattr(state, "values", None):
            return []
        msgs = state.values.get("messages", [])
        out: list[dict] = []
        for m in msgs:
            role = (
                "user"
                if isinstance(m, HumanMessage)
                else "assistant"
                if isinstance(m, AIMessage)
                else "tool"
                if isinstance(m, ToolMessage)
                else "system"
            )
            out.append({"role": role, "content": getattr(m, "content", "")})
        return out
