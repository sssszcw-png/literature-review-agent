"""Application settings loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_env_file() -> None:
    """Load .env file once. Called from get_settings(), not at import time."""
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek API
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_request_timeout: int = 120

    # Search APIs
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_rate_limit: float = 1.0
    arxiv_base_url: str = "https://export.arxiv.org/api/query"
    search_max_results_per_query: int = 20
    search_timeout: int = 30

    # PDF Processing
    pdf_cache_dir: Path = Path("cache/papers")
    pdf_download_dir: Path = Path("cache/pdfs")
    pdf_parse_timeout: int = 120
    pdf_max_size_mb: int = 50

    # Agent Behavior
    default_max_rounds: int = 5
    saturation_threshold: float = 0.7
    max_no_improvement_rounds: int = 2
    max_concurrent_searches: int = 5
    max_concurrent_chapters: int = 5
    max_papers_per_round: int = 50

    # Output
    default_output_dir: Path = Path("reports")

    # Retry
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0
    retry_max_delay: float = 60.0

    # Checkpoint
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_enabled: bool = True

    # Prompts
    prompts_dir: Path = Path("prompts")

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton. .env is loaded once on first call."""
    _load_env_file()
    return Settings()
