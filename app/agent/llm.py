"""Ollama LLM factory."""
from __future__ import annotations

from app.agent.tool_calling_ollama import ToolCallingChatOllama
from app.config import get_settings


def build_llm() -> ToolCallingChatOllama:
    """Create a ChatOllama instance with reliable tool-call recovery.

    See :mod:`app.agent.tool_calling_ollama` for why we wrap rather than
    use ``ChatOllama`` directly.
    """
    s = get_settings()
    return ToolCallingChatOllama(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        temperature=s.ollama_temperature,
        num_ctx=s.ollama_num_ctx,
    )
