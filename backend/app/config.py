"""Backend configuration — 12-factor, env-overridable."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = REPO_ROOT / "data"

class Settings:
    app_name: str = "DataPilot API"
    version: str = "2.0.0-fullstack"
    api_prefix: str = "/api/v1"
    max_file_mb: int = int(os.getenv("MAX_FILE_MB", "250"))
    max_file_bytes: int = max_file_mb * 1024 * 1024
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL", "3600"))
    max_model_rows: int = int(os.getenv("MAX_MODEL_ROWS", "100000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    data_dir: Path = DATA_DIR

@lru_cache()
def get_settings() -> Settings:
    return Settings()
