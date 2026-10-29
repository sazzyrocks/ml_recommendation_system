"""
backend/routes/recommendations.py
---------------------------------
Personalized recommendations endpoint:
- GET /recommendations/me?k=20: Generates hybrid recommendations with fold-in
  adaptive personalization and explainability reasons based on user's DB ratings.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.ml_service import ml_service
from backend.models.entities import Movie, Rating, User
from backend.schemas import RecommendationItemResponse, RecommendationsResponse
from backend.security import get_current_user

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get(
    "/me",
    response_model=RecommendationsResponse,
    summary="Get personalized movie recommendations for the current user",
)
async def get_my_recommendations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    k: int = Query(20, ge=1, le=50, description="Number of recommendations to return"),
) -> RecommendationsResponse:
    """
    Generate top-K hybrid recommendations for the authenticated user:
    - Queries the user's ratings stored in the database.
    - Applies instant fold-in least-squares update to personalize recommendations.
    - For cold-start users (0 ratings), falls back to genre onboarding / popularity baseline.
    - Annotates each recommendation with a clear explainability reason.
    """
    # 1. Fetch user's ratings from database
    query = (
        select(Movie.id, Movie.ml_movie_id, Rating.rating)
        .join(Movie, Rating.movie_id == Movie.id)
        .where(Rating.user_id == current_user.id)
    )
    user_ratings_rows = (await db.execute(query)).all()
    user_rated_db_ids = {int(r[0]) for r in user_ratings_rows}
    user_ratings_db: dict[int, float] = {int(r[1]): float(r[2]) for r in user_ratings_rows}

    # 2. Generate recommendations from MLService
    raw_recs = ml_service.recommend_for_user(user_ratings_db=user_ratings_db, k=k)

    if not raw_recs:
        # DB fallback to catalog movies excluding already rated movies
        fallback_query = select(Movie)
        if user_rated_db_ids:
            fallback_query = fallback_query.where(Movie.id.not_in(user_rated_db_ids))
        fallback_query = fallback_query.limit(k)
        fb_movies = (await db.execute(fallback_query)).scalars().all()
        return RecommendationsResponse(
            items=[
                RecommendationItemResponse(
                    movie_id=m.id,
                    title=m.title,
                    year=m.year,
                    genres=m.genres,
                    poster_path=m.poster_path,
                    score=0.5,
                    reason="Trending across CineMatch viewers",
                    top_source="popularity",
                )
                for m in fb_movies
            ],
            user_id=current_user.id,
            count=len(fb_movies),
        )

    # 3. Resolve DB movie records for the recommended items
    # raw_recs contains items with 'movieId' (MovieLens ID) or 'item_idx'
    mids_to_fetch = [r["movieId"] for r in raw_recs if r.get("movieId") is not None]
    db_movies = (
        (await db.execute(select(Movie).where(Movie.ml_movie_id.in_(mids_to_fetch))))
        .scalars()
        .all()
    )
    mid_to_db_movie = {m.ml_movie_id: m for m in db_movies}

    items: list[RecommendationItemResponse] = []
    for r in raw_recs:
        mid = r.get("movieId")
        db_m = mid_to_db_movie.get(mid)
        if db_m and db_m.id in user_rated_db_ids:
            continue

        movie_id = db_m.id if db_m else int(r.get("item_idx", 0))
        if movie_id in user_rated_db_ids:
            continue
        title = db_m.title if db_m else str(r.get("title", ""))
        year = db_m.year if db_m else r.get("year")
        genres = db_m.genres if db_m else r.get("genres", [])
        poster = db_m.poster_path if db_m else None

        items.append(
            RecommendationItemResponse(
                movie_id=movie_id,
                title=title,
                year=year,
                genres=genres,
                poster_path=poster,
                score=round(float(r.get("score", 0.0)), 4),
                reason=str(r.get("reason", "Recommended for you")),
                top_source=str(r.get("top_source", "hybrid")),
            )
        )

    return RecommendationsResponse(
        items=items[:k],
        user_id=current_user.id,
        count=len(items[:k]),
    )
