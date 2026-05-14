"""Centralised settings loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All values can be overridden via environment variables or a local
    ``.env`` file (see ``.env.example``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_embed_model: str = "nomic-embed-text:latest"
    ollama_temperature: float = 0.2
    ollama_num_ctx: int = 8192

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "*"

    # --- Workspace ---
    workspace_root: Path = Field(default_factory=lambda: Path.cwd())

    # --- Storage ---
    data_dir: Path = Path("./data")
    sqlite_path: Path = Path("./data/assistant.sqlite")
    chroma_path: Path = Path("./data/chroma")

    # --- Terminal safety ---
    terminal_allowlist: str = (
        "git,python,pytest,pip,ls,dir,cat,type,grep,rg,echo,pwd,cd,"
        "docker,kubectl,npm,node"
    )
    terminal_timeout_seconds: int = 60
    terminal_max_output_bytes: int = 200_000

    # --- Agent ---
    agent_max_iterations: int = 12
    agent_recursion_limit: int = 40

    # ---------- Helpers ----------

    @field_validator("workspace_root", "data_dir", "sqlite_path", "chroma_path")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @property
    def cors_origins_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def terminal_allowlist_set(self) -> set[str]:
        return {x.strip().lower() for x in self.terminal_allowlist.split(",") if x.strip()}

    def ensure_dirs(self) -> None:
        """Create writable directories (data, chroma, sqlite parent)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor used everywhere in the codebase."""
    return Settings()
