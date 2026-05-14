# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps:
#  - git: needed by GitPython and the git MCP server
#  - build-essential: chroma's onnxruntime / native deps on some bases
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY mcp_servers ./mcp_servers
COPY tests ./tests

# Default Ollama URL points to host's Ollama; override at runtime if needed.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/app/data \
    SQLITE_PATH=/app/data/assistant.sqlite \
    CHROMA_PATH=/app/data/chroma

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
