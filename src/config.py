"""Application configuration using Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central application settings, loaded from environment / .env file."""

    # ── DeepSeek LLM ──
    llm_api_key: str = Field(default="", description="DeepSeek API key")
    llm_base_url: str = Field(
        default="https://api.deepseek.com", description="DeepSeek API base URL"
    )
    llm_model: str = Field(default="deepseek-chat", description="DeepSeek model name")
    llm_temperature: float = Field(default=0.1, description="LLM temperature for SAR generation")
    llm_max_tokens: int = Field(default=8192, description="LLM max output tokens")

    # ── Compliance ──
    compliance_score_threshold: float = Field(
        default=0.75, description="Minimum compliance score to pass validation"
    )
    max_iterations: int = Field(
        default=3, description="Maximum feedback iteration rounds"
    )

    # ── Paths ──
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    samples_dir: Path = Field(default=PROJECT_ROOT / "data" / "samples")
    prompts_dir: Path = Field(default=PROJECT_ROOT / "prompts")
    db_dir: Path = Field(default=PROJECT_ROOT / "data" / "db")

    # ── Checkpoints ──
    checkpoint_db: str = Field(
        default="checkpoints.db", description="SQLite checkpoint database filename"
    )

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
