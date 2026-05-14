"""ChromaDB-backed vector memory for repository content.

Provides:
    * ``RepoIndexer.index_workspace()`` — walk the workspace, chunk and
      embed text/code files into Chroma.
    * ``RepoIndexer.search(query, k)``  — semantic search over the index.

Embeddings come from Ollama (``nomic-embed-text`` by default), so the
whole pipeline stays 100 % local.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

import chromadb
from langchain_ollama import OllamaEmbeddings

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb",
    ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt",
    ".m", ".mm", ".scala", ".sh", ".ps1", ".bat", ".yaml", ".yml",
    ".toml", ".json", ".xml", ".html", ".css", ".scss", ".md", ".rst",
    ".txt", ".ini", ".cfg", ".env",
}

IGNORE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".idea", ".vscode", "data", "chroma",
}

CHUNK_SIZE = 1500          # characters
CHUNK_OVERLAP = 200
MAX_FILE_BYTES = 200_000   # skip enormous generated/min files
COLLECTION_NAME = "workspace"


def _chunk_text(text: str) -> List[str]:
    """Naive but robust character-level chunker with overlap."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _file_id(path: Path, idx: int) -> str:
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{h}-{idx}"


class RepoIndexer:
    """Wraps a Chroma collection and an Ollama embedder."""

    def __init__(self) -> None:
        s = get_settings()
        s.ensure_dirs()
        self._client = chromadb.PersistentClient(path=str(s.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = OllamaEmbeddings(
            base_url=s.ollama_base_url,
            model=s.ollama_embed_model,
        )

    # ---------- Indexing ----------

    def index_workspace(self) -> dict:
        s = get_settings()
        root = s.workspace_root
        files_indexed = 0
        chunks_indexed = 0

        # Reset for a clean re-index. Production callers can replace this
        # with an incremental update strategy if desired.
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        batch_texts: list[str] = []
        batch_ids: list[str] = []
        batch_meta: list[dict] = []

        def flush() -> None:
            nonlocal batch_texts, batch_ids, batch_meta, chunks_indexed
            if not batch_texts:
                return
            vectors = self._embed.embed_documents(batch_texts)
            self._collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_meta,
                embeddings=vectors,
            )
            chunks_indexed += len(batch_texts)
            batch_texts, batch_ids, batch_meta = [], [], []

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_EXTS:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not text.strip():
                continue

            rel = str(path.relative_to(root))
            for i, chunk in enumerate(_chunk_text(text)):
                cleaned = re.sub(r"\s+", " ", chunk).strip()
                if not cleaned:
                    continue
                batch_texts.append(chunk)
                batch_ids.append(_file_id(path, i))
                batch_meta.append({"path": rel, "chunk": i})
                if len(batch_texts) >= 32:
                    flush()
            files_indexed += 1

        flush()
        logger.info(
            "vector.index",
            files=files_indexed,
            chunks=chunks_indexed,
            root=str(root),
        )
        return {
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "workspace": str(root),
        }

    # ---------- Querying ----------

    def search(self, query: str, k: int = 6) -> list[dict]:
        if self._collection.count() == 0:
            return []
        q_vec = self._embed.embed_query(query)
        res = self._collection.query(
            query_embeddings=[q_vec],
            n_results=max(1, min(k, 20)),
        )
        out: list[dict] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "path": meta.get("path"),
                    "chunk": meta.get("chunk"),
                    "score": round(1.0 - float(dist), 4),
                    "snippet": doc[:600],
                }
            )
        return out

    def stats(self) -> dict:
        return {
            "collection": COLLECTION_NAME,
            "vector_count": self._collection.count(),
            "embedding_model": get_settings().ollama_embed_model,
        }
