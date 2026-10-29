"""
backend/main.py
---------------
FastAPI application entry point for CineMatch AI.
Loads ML artifacts once during lifespan startup and exposes production REST endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.ml_service import ml_service
from backend.routes import (
    analytics_router,
    auth_router,
    movies_router,
    ratings_router,
    recommendations_router,
    watchlist_router,
)

settings = get_settings()

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager:
    Loads heavy ML artifacts (ImplicitALS, ItemKNN, Content models, Embeddings)
    once into memory at startup to avoid per-request latency.
    """
    logger.info("Lifespan: Loading ML artifacts into memory...")
    try:
        await asyncio.to_thread(ml_service.load_artifacts)
        logger.info("Lifespan: ML artifacts successfully initialized.")
    except Exception as e:
        logger.warning(f"Lifespan: Could not load ML artifacts: {e}. Proceeding in basic mode.")

    yield

    logger.info("Lifespan: Shutting down CineMatch backend...")


app = FastAPI(
    title="CineMatch AI",
    description="Portfolio-grade personalized movie recommendation backend powered by ML and vector search.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Restricted strictly to configured frontend origins from .env
origins = settings.cors_origins or [settings.frontend_origin]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler for Consistent Errors ───────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Let HTTPException pass through unchanged
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    logger.exception(f"Unhandled error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"], summary="Liveness & Health Probe")
async def health() -> dict[str, str]:
    """Simple liveness probe. Returns 200 when the server is healthy."""
    return {
        "status": "ok",
        "environment": settings.environment,
        "models_loaded": "true" if ml_service.hybrid_model is not None else "false",
    }


# ── Register Routers ──────────────────────────────────────────────────────────
# Both prefixed with /api/v1 and top-level for convenience
for prefix in ["/api/v1", ""]:
    app.include_router(auth_router, prefix=prefix)
    app.include_router(movies_router, prefix=prefix)
    app.include_router(ratings_router, prefix=prefix)
    app.include_router(watchlist_router, prefix=prefix)
    app.include_router(recommendations_router, prefix=prefix)
    app.include_router(analytics_router, prefix=prefix)
