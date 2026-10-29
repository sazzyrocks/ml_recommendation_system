"""
backend/config.py
-----------------
Centralised application settings loaded from environment variables / .env file.
Pydantic-Settings validates and casts every value at startup.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://cinematch:changeme@localhost:5432/cinematch_db"
    test_database_url: str = "sqlite+aiosqlite:///./test_cinematch.db"

    # ── JWT ───────────────────────────────────────────────────────────────
    secret_key: str = (
        "cinematch_super_secret_jwt_key_phase10_portfolio_grade_2026_dev_mode_abcdef0123456789"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── TMDB ──────────────────────────────────────────────────────────────
    tmdb_api_key: str = ""
    tmdb_base_url: str = "https://api.themoviedb.org/3"

    # ── App & Security ────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"
    frontend_origin: str = "http://localhost:5173"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of Settings (reads .env once)."""
    return Settings()
