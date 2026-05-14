"""FastAPI application entrypoint.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import AssistantAgent
from app.api.routes import router as api_router
from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.mcp_client.manager import MCPManager
from app.memory.tools import build_memory_tools
from app.memory.vector_store import RepoIndexer

configure_logging()
logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    logger.info(
        "startup",
        workspace=str(settings.workspace_root),
        model=settings.ollama_model,
        ollama=settings.ollama_base_url,
    )

    mcp_manager = MCPManager()
    await mcp_manager.start()

    indexer = RepoIndexer()

    tools = mcp_manager.tools + build_memory_tools(indexer)
    agent = AssistantAgent(tools=tools)
    await agent.start()

    app.state.mcp = mcp_manager
    app.state.indexer = indexer
    app.state.agent = agent

    try:
        yield
    finally:
        logger.info("shutdown")
        await agent.stop()
        await mcp_manager.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="MCP-Powered Local AI Engineering Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)
