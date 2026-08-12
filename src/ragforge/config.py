from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAGForge"
    environment: str = "production"
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    app_api_token: str | None = Field(default=None, alias="APP_API_TOKEN")

    default_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    native_search_model: str = "gemini-2.5-flash-lite"

    max_upload_mb: int = 20
    max_archive_files: int = 30
    max_archive_uncompressed_mb: int = 60
    max_chunks_per_session: int = 5000
    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 250
    top_k_dense: int = 12
    top_k_sparse: int = 12
    top_k_final: int = 6
    session_ttl_minutes: int = 120
    queries_per_hour_per_ip: int = 40
    cache_ttl_seconds: int = 600
    llm_max_retries: int = 2
    allow_server_api_key: bool = True
    enable_native_google_search: bool = True

    data_dir: Path = Path("/tmp/ragforge")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
