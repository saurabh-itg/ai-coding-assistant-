"""ChatOllama subclass that recovers tool calls from message content.

Background
----------
Some Ollama models (notably ``qwen2.5-coder``) emit tool calls as raw
JSON in the assistant *content* instead of populating the structured
``tool_calls`` field that LangChain / LangGraph expect. The shape is
typically:

    {"name": "list_files", "arguments": {"path": "."}}

LangGraph's ReAct agent only dispatches tools when an ``AIMessage``
arrives with a non-empty ``tool_calls`` list, so out of the box the
agent treats the JSON above as a final answer.

This wrapper post-processes every assistant message produced by
``ChatOllama`` (sync, async, and streaming) and converts that
content-JSON into a proper ``tool_calls`` entry — making models with
weak native tool-calling templates behave the same as ones with strong
ones.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama


_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Strip ```json … ``` fences if the model wrapped its tool call."""
    if "```" not in text:
        return text
    return _FENCE_RE.sub("", text).strip()


def _is_tool_call_obj(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters"))
    return isinstance(name, str) and isinstance(args, dict)


def _to_tool_call(obj: dict) -> dict:
    args = obj.get("arguments", obj.get("parameters"))
    return {
        "name": obj["name"],
        "args": args,
        "id": "call_" + uuid.uuid4().hex[:12],
        "type": "tool_call",
    }


def _scan_json_objects(text: str) -> List[dict]:
    """Scan ``text`` for one or more JSON objects.

    Uses :class:`json.JSONDecoder.raw_decode` to peel objects off
    sequentially. Tolerates whitespace, prose, and ``,``/``;`` separators
    between objects.
    """
    decoder = json.JSONDecoder()
    out: List[dict] = []
    i = 0
    n = len(text)
    while i < n:
        # Skip to the next plausible start of a JSON object.
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def _extract_tool_call(content: str) -> Optional[dict]:
    """Parse the *first* tool call from ``content``. Used by tests."""
    calls = _extract_tool_calls(content)
    return calls[0] if calls else None


def _extract_tool_calls(content: str) -> List[dict]:
    """Parse one or more ``{"name": ..., "arguments": {...}}`` objects."""
    if not content or not content.strip():
        return []
    body = _strip_fences(content).strip()
    objects = _scan_json_objects(body)
    if not objects:
        return []
    valid = [o for o in objects if _is_tool_call_obj(o)]
    if not valid:
        return []
    return [_to_tool_call(o) for o in valid]


def _rewrite(msg: AIMessage) -> AIMessage:
    """Return a new ``AIMessage`` with tool_calls populated when possible."""
    if msg.tool_calls:
        return msg
    content = msg.content if isinstance(msg.content, str) else ""
    calls = _extract_tool_calls(content)
    if not calls:
        return msg
    return AIMessage(
        content="",
        tool_calls=calls,
        response_metadata=msg.response_metadata,
        id=msg.id,
        usage_metadata=getattr(msg, "usage_metadata", None),
    )


class ToolCallingChatOllama(ChatOllama):
    """ChatOllama with content-to-tool-call recovery for weak templates.

    Drop-in replacement: passes ``bind_tools`` straight through and
    rewrites the produced :class:`AIMessage` after generation.

    Note on streaming
    -----------------
    LangGraph's ReAct agent calls the LLM through the streaming
    interface during ``astream_events``. Tool calls would slip past
    our post-processor in that path, so we disable native streaming
    entirely. ``BaseChatModel.disable_streaming = True`` makes
    ``_astream`` / ``_stream`` round-trip through ``_agenerate`` /
    ``_generate``, then yield a single aggregated chunk — exactly
    what we want.
    """

    disable_streaming: bool = True

    def _generate(  # type: ignore[override]
        self, messages, stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._rewrite_result(result)

    async def _agenerate(  # type: ignore[override]
        self, messages, stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        result = await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return self._rewrite_result(result)

    @staticmethod
    def _rewrite_result(result: ChatResult) -> ChatResult:
        new_gens = []
        for gen in result.generations:
            msg = gen.message
            if isinstance(msg, AIMessage):
                new_gens.append(
                    ChatGeneration(
                        message=_rewrite(msg),
                        generation_info=gen.generation_info,
                    )
                )
            else:
                new_gens.append(gen)
        return ChatResult(generations=new_gens, llm_output=result.llm_output)
