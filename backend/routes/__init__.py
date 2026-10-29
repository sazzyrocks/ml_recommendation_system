"""
backend/routes/__init__.py
"""

from backend.routes.analytics import router as analytics_router
from backend.routes.auth import router as auth_router
from backend.routes.movies import router as movies_router
from backend.routes.ratings import router as ratings_router
from backend.routes.recommendations import router as recommendations_router
from backend.routes.watchlist import router as watchlist_router

__all__ = [
    "auth_router",
    "movies_router",
    "ratings_router",
    "watchlist_router",
    "recommendations_router",
    "analytics_router",
]
