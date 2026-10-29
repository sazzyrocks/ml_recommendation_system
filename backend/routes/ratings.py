"""
backend/routes/ratings.py
-------------------------
User ratings management endpoints:
- POST /ratings: Submit new rating for a movie (0.5 to 5.0)
- PUT /ratings/{movie_id}: Update an existing rating
- DELETE /ratings/{movie_id}: Delete a rating
- GET /ratings/me: List all ratings submitted by the current user
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models.entities import Movie, Rating, User
from backend.schemas import RatingCreateRequest, RatingResponse, RatingUpdateRequest
from backend.security import get_current_user

router = APIRouter(prefix="/ratings", tags=["ratings"])


@router.post(
    "",
    response_model=RatingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a movie rating",
)
async def create_rating(
    payload: RatingCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Rating:
    """Submit a rating between 0.5 and 5.0 stars for a movie."""
    # Verify movie exists
    movie = (
        await db.execute(select(Movie).where(Movie.id == payload.movie_id))
    ).scalar_one_or_none()
    if movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")

    # Check if user already rated this movie
    query = select(Rating).where(
        Rating.user_id == current_user.id, Rating.movie_id == payload.movie_id
    )
    existing = (await db.execute(query)).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already rated this movie. Use PUT to update your rating.",
        )

    rating_entry = Rating(
        user_id=current_user.id,
        movie_id=payload.movie_id,
        rating=payload.rating,
    )
    db.add(rating_entry)
    await db.commit()

    # Re-fetch with loaded movie relationship
    res = await db.execute(
        select(Rating).options(selectinload(Rating.movie)).where(Rating.id == rating_entry.id)
    )
    return res.scalar_one()


@router.put(
    "/{movie_id}",
    response_model=RatingResponse,
    summary="Update an existing movie rating",
)
async def update_rating_by_path(
    movie_id: int,
    payload: RatingUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Rating:
    """Update a user's existing rating for a movie by ID."""
    return await _do_update_rating(movie_id, payload.rating, current_user.id, db)


@router.put(
    "",
    response_model=RatingResponse,
    summary="Update a movie rating via body",
)
async def update_rating(
    payload: RatingCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Rating:
    """Update a user's existing rating for a movie via request body."""
    return await _do_update_rating(payload.movie_id, payload.rating, current_user.id, db)


async def _do_update_rating(
    movie_id: int, new_rating: float, user_id: int, db: AsyncSession
) -> Rating:
    query = select(Rating).where(Rating.user_id == user_id, Rating.movie_id == movie_id)
    existing = (await db.execute(query)).scalar_one_or_none()
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rating not found.")

    existing.rating = new_rating
    await db.commit()

    res = await db.execute(
        select(Rating).options(selectinload(Rating.movie)).where(Rating.id == existing.id)
    )
    return res.scalar_one()


@router.delete(
    "/{movie_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a movie rating by movie ID",
)
async def delete_rating_by_path(
    movie_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Delete a user's rating for a movie by ID."""
    await _do_delete_rating(movie_id, current_user.id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a movie rating via query parameter",
)
async def delete_rating(
    movie_id: int = Query(..., description="ID of the movie to remove rating for"),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> Response:
    """Delete a user's rating for a movie via query parameter."""
    await _do_delete_rating(movie_id, current_user.id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _do_delete_rating(movie_id: int, user_id: int, db: AsyncSession) -> None:
    query = select(Rating).where(Rating.user_id == user_id, Rating.movie_id == movie_id)
    existing = (await db.execute(query)).scalar_one_or_none()
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rating not found.")

    await db.delete(existing)
    await db.commit()


@router.get(
    "/me",
    response_model=list[RatingResponse],
    summary="List all ratings by current user",
)
async def get_my_ratings(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Rating]:
    """Retrieve all ratings submitted by the authenticated user."""
    query = (
        select(Rating)
        .options(selectinload(Rating.movie))
        .where(Rating.user_id == current_user.id)
        .order_by(Rating.created_at.desc())
    )
    items = (await db.execute(query)).scalars().all()
    return list(items)
