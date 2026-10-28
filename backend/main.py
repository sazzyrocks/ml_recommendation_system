"""
backend/main.py
FastAPI application entry point.
Routers are registered here; they will be filled in later phases.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings

settings = get_settings()

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CineMatch AI",
    description="Personalized movie recommendations powered by ML.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# During development allow all origins. Lock this down before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Simple liveness probe. Returns 200 when the server is up."""
    return {"status": "ok", "environment": settings.environment}


# ── Routers will be imported here in later phases ─────────────────────────────
# from backend.routes import auth, movies, recommendations, ratings, watchlist, analytics
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
# ... etc.
