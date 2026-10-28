"""
backend/config.py
Centralised settings loaded from environment variables / .env file.
Pydantic-Settings validates and casts every value at startup.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tuneable parameters for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://cinematch:changeme@localhost:5432/cinematch_db"

    # ── JWT ───────────────────────────────────────────────────────────────
    secret_key: str = "REPLACE_WITH_64_CHAR_HEX_STRING"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── TMDB ──────────────────────────────────────────────────────────────
    tmdb_api_key: str = ""
    tmdb_base_url: str = "https://api.themoviedb.org/3"

    # ── App ───────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of Settings (reads .env once)."""
    return Settings()
