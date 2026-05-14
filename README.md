# MCP-Powered Local AI Engineering Assistant

A fully local AI engineering assistant that can read your codebase, analyse
git history, and run a curated set of terminal commands — all driven by a
local LLM via [Ollama](https://ollama.ai) and orchestrated through three
MCP (Model Context Protocol) connectors.

```
            ┌──────────────┐
 User ─────▶│  FastAPI UI  │
            └──────┬───────┘
                   ▼
            ┌──────────────┐         ┌──────────────────┐
            │  LangGraph   │◀── LLM ─│  Ollama (local)  │
            │  ReAct Agent │         │  qwen2.5-coder   │
            └──────┬───────┘         └──────────────────┘
                   │ MCP tools
        ┌──────────┼─────────┬──────────┐
        ▼          ▼         ▼          ▼
   filesystem    git    terminal    vector mem
    MCP svr   MCP svr   MCP svr     (Chroma + nomic-embed)
```

The assistant can answer questions like:

* *"Explain this repository."*
* *"Find risky / TODO code."*
* *"Summarise the last 10 commits."*
* *"Which files relate to authentication?"*
* *"Why are the tests failing?"*

---

## Features

* **3 MCP servers** (stdio transport, sandboxed to a workspace):
  * `filesystem` — `list_files`, `read_file`, `search_code`, `file_stats`,
    `list_directory_tree`
  * `git` — `recent_commits`, `changed_files`, `branch_diff`, `blame_file`,
    `git_status`, `current_branch`
  * `terminal` — `run_command` (allow-listed only), `list_allowed`
* **LangGraph ReAct agent** that decides which tools to call.
* **Persistent chat memory** in SQLite via `AsyncSqliteSaver`.
* **Vector memory** in ChromaDB, embedded with `nomic-embed-text` (Ollama).
  The agent can call `search_docs` to do RAG over the workspace.
* **Streaming chat UI** at `/` (single-page, no build step).
* **Production-ready** — structured logging (structlog), CORS,
  health checks, env-based config (pydantic-settings), Dockerfile,
  docker-compose, allow-listed terminal, path-traversal protection,
  command timeouts, and pytest test suite.

---

## Prerequisites

1. **Python 3.11+** (3.12 recommended).
2. **Git** on the PATH.
3. **Ollama** running locally with the models you intend to use:
   ```powershell
   ollama pull qwen2.5-coder:7b
   ollama pull nomic-embed-text
   ```
   Verify with `ollama list`.

---

## Quick start (Windows / PowerShell)

```powershell
# 1. install deps + create .env
./run.ps1 setup

# 2. edit .env if you want a different model or workspace
notepad .env

# 3. start the server
./run.ps1 start

# 4. open the UI
start http://localhost:8000
```

## Quick start (Linux / macOS)

```bash
chmod +x run.sh
./run.sh setup
./run.sh start
open http://localhost:8000
```

## Quick start (Docker)

```bash
# Set the host path you want the assistant to read.
# On Windows PowerShell:
$env:WORKSPACE_HOST_PATH = "C:/path/to/your/repo"

docker compose up --build
```

The assistant talks to Ollama on the **host** through
`host.docker.internal`, so make sure Ollama is listening on
`0.0.0.0:11434` (or set `OLLAMA_HOST=0.0.0.0` before starting Ollama).

---

## Configuration (`.env`)

| Variable                    | Default                         | Purpose                                    |
| --------------------------- | ------------------------------- | ------------------------------------------ |
| `OLLAMA_BASE_URL`           | `http://localhost:11434`        | Ollama HTTP endpoint                       |
| `OLLAMA_MODEL`              | `qwen2.5-coder:7b`              | Chat model                                 |
| `OLLAMA_EMBED_MODEL`        | `nomic-embed-text:latest`       | Embedding model                            |
| `OLLAMA_TEMPERATURE`        | `0.2`                           | Sampling temperature                       |
| `OLLAMA_NUM_CTX`            | `8192`                          | Context window                             |
| `WORKSPACE_ROOT`            | *cwd*                           | Repo the assistant is sandboxed to         |
| `TERMINAL_ALLOWLIST`        | `git,python,pytest,…`           | Comma-separated executables allowed        |
| `TERMINAL_TIMEOUT_SECONDS`  | `60`                            | Hard kill timeout                          |
| `TERMINAL_MAX_OUTPUT_BYTES` | `200000`                        | Output truncation cap                      |
| `AGENT_RECURSION_LIMIT`     | `40`                            | Max LangGraph steps per turn               |
| `LOG_LEVEL`                 | `INFO`                          | Log verbosity                              |
| `CORS_ORIGINS`              | `*`                             | CSV list, or `*`                           |

---

## API surface

* `GET  /api/health` — service status, Ollama reachability, tool count
* `GET  /api/tools` — all MCP tool names + JSON Schemas
* `POST /api/chat` — body `{ "message", "thread_id"?, "stream"? }`
  * `stream: true` (default) → Server-Sent Events
    (`event: token | tool_start | tool_end | final | done`)
  * `stream: false` → JSON `{ thread_id, answer, tool_calls }`
* `GET  /api/history/{thread_id}` — conversation transcript
* `POST /api/index` — re-index the workspace into the vector store
* `GET  /api/index/stats` — vector count, embedding model

A minimal web UI is served at `/`.

---

## Safety

* **Workspace sandbox** — every filesystem / git tool resolves user paths
  with `Path.resolve()` and rejects anything outside `WORKSPACE_ROOT`.
* **Terminal allow-list** — `run_command` parses the command without a
  shell (`shell=False`), checks the head executable against an
  allow-list, runs with a timeout and hard byte cap, and refuses any
  `cwd` that escapes the workspace.
* **Read-only by default** — the only write surface in the box is whatever
  the user explicitly enables via `TERMINAL_ALLOWLIST`.
* **Local LLM** — no calls to external APIs.

---

## Tests

```powershell
./run.ps1 test
# or
pytest -q
```

The test suite covers:
* path-traversal sandboxing (`test_safety.py`)
* filesystem tool behaviour (`test_filesystem_server.py`)
* terminal allow-list / cwd protection (`test_terminal_server.py`)
* settings parsing (`test_config.py`)

---

## Layout

```
app/
  main.py             FastAPI app + lifespan (boots MCP, agent, indexer)
  config.py           pydantic-settings
  logging_config.py   structlog setup
  agent/
    graph.py          LangGraph ReAct agent + SQLite checkpointer
    llm.py            ChatOllama factory
    prompts.py        system prompt
  api/
    routes.py         /chat (SSE), /health, /tools, /index, /history
    schemas.py        pydantic IO models
  mcp_client/
    manager.py        MultiServerMCPClient lifecycle wrapper
  memory/
    vector_store.py   Chroma + Ollama embeddings
    tools.py          search_docs LangChain tool
  static/index.html   chat UI

mcp_servers/
  _safety.py          shared path-sandbox helpers
  filesystem_server.py
  git_server.py
  terminal_server.py

tests/                pytest suite
```
