"""
backend/routes/watchlist.py
---------------------------
Watchlist management endpoints:
- GET /watchlist: Retrieve current user's saved movies
- POST /watchlist: Add a movie to user's watchlist
- DELETE /watchlist/{movie_id}: Remove a movie from user's watchlist
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models.entities import Movie, User, Watchlist
from backend.schemas import WatchlistCreateRequest, WatchlistResponse
from backend.security import get_current_user

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.get(
    "",
    response_model=list[WatchlistResponse],
    summary="List current user's watchlist",
)
async def get_watchlist(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Watchlist]:
    """Retrieve all movies saved in the authenticated user's watchlist."""
    query = (
        select(Watchlist)
        .options(selectinload(Watchlist.movie))
        .where(Watchlist.user_id == current_user.id)
        .order_by(Watchlist.added_at.desc())
    )
    items = (await db.execute(query)).scalars().all()
    return list(items)


@router.post(
    "",
    response_model=WatchlistResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a movie to watchlist",
)
async def add_to_watchlist(
    payload: WatchlistCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Watchlist:
    """Add a movie to the authenticated user's watchlist."""
    # Check if movie exists
    movie = (
        await db.execute(select(Movie).where(Movie.id == payload.movie_id))
    ).scalar_one_or_none()
    if movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")

    # Check if already in watchlist
    query = select(Watchlist).where(
        Watchlist.user_id == current_user.id, Watchlist.movie_id == payload.movie_id
    )
    existing = (await db.execute(query)).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Movie is already in your watchlist.",
        )

    entry = Watchlist(user_id=current_user.id, movie_id=payload.movie_id)
    db.add(entry)
    await db.commit()

    res = await db.execute(
        select(Watchlist).options(selectinload(Watchlist.movie)).where(Watchlist.id == entry.id)
    )
    return res.scalar_one()


@router.delete(
    "/{movie_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove a movie from watchlist by movie ID",
)
async def remove_from_watchlist_by_path(
    movie_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Remove a movie from the authenticated user's watchlist by ID."""
    await _do_remove_watchlist(movie_id, current_user.id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove a movie from watchlist via query param",
)
async def remove_from_watchlist(
    movie_id: int = Query(..., description="ID of the movie to remove from watchlist"),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> Response:
    """Remove a movie from the authenticated user's watchlist via query parameter."""
    await _do_remove_watchlist(movie_id, current_user.id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _do_remove_watchlist(movie_id: int, user_id: int, db: AsyncSession) -> None:
    query = select(Watchlist).where(Watchlist.user_id == user_id, Watchlist.movie_id == movie_id)
    existing = (await db.execute(query)).scalar_one_or_none()
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not in watchlist.")

    await db.delete(existing)
    await db.commit()
