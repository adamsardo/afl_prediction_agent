from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AFL_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AFL Prediction Agent"
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/afl_agent"
    )
    api_prefix: str = "/api"
    workspace_root: Path = ROOT_DIR
    default_llm_provider: str = "heuristic"
    default_analyst_model: str = "heuristic-analyst-v1"
    default_case_model: str = "heuristic-case-v1"
    default_final_model: str = "heuristic-final-v1"
    prompts_dir: Path = ROOT_DIR / "config" / "prompts"
    run_config_dir: Path = ROOT_DIR / "config" / "run_configs"
    model_artifact_dir: Path = ROOT_DIR / "data" / "artifacts"
    codex_bin: str = "codex"
    codex_startup_timeout_seconds: float = 15.0
    codex_turn_timeout_seconds: float = 120.0
    fitzroy_rscript_bin: str = "Rscript"
    fitzroy_timeout_seconds: float = 120.0
    source_http_timeout_seconds: float = 20.0
    source_retry_attempts: int = 2
    odds_api_key: str | None = None
    odds_au_bookmakers: str | None = None
    squiggle_benchmark_source: str = "aggregate"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
