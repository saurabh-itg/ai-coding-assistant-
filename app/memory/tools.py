"""LangChain tools backed by the vector store, so the agent can do RAG."""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.memory.vector_store import RepoIndexer


class _SearchDocsArgs(BaseModel):
    query: str = Field(..., description="Natural-language query.")
    k: int = Field(6, description="Max number of results to return (1-20).")


def build_memory_tools(indexer: RepoIndexer) -> list:
    """Wrap the indexer's search method as a LangChain tool."""

    def _search_docs(query: str, k: int = 6) -> dict:
        hits = indexer.search(query, k=k)
        return {"query": query, "hits": hits}

    return [
        StructuredTool.from_function(
            func=_search_docs,
            name="search_docs",
            description=(
                "Semantic search over the indexed workspace. "
                "Use this when you need to find files / passages related to "
                "a concept (e.g. 'authentication middleware', 'database "
                "connection retry logic') without knowing exact filenames."
            ),
            args_schema=_SearchDocsArgs,
        )
    ]
