"""
backend/routes/movies.py
------------------------
Catalog, search, and similarity endpoints:
- GET /movies: Paginated catalog listing with genre and year filtering
- GET /movies/{id}: Movie details with average rating and rating counts
- GET /movies/{id}/similar: Content-based nearest neighbor recommendations
- GET /search?q=: Semantic natural-language search via 384-dim dense embeddings
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.ml_service import ml_service
from backend.models.entities import Movie, Rating
from backend.schemas import (
    MovieDetailResponse,
    MovieResponse,
    PaginatedMoviesResponse,
    SearchResultResponse,
    SimilarMovieResponse,
)

router = APIRouter(tags=["movies"])


@router.get(
    "/movies",
    response_model=PaginatedMoviesResponse,
    summary="List movies with pagination and filters",
)
async def list_movies(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    genre: str | None = Query(None, description="Filter by genre"),
    year: int | None = Query(None, description="Filter by exact release year"),
    year_min: int | None = Query(None, description="Filter by min release year"),
    year_max: int | None = Query(None, description="Filter by max release year"),
    sort: Literal["id", "popular"] = Query("id", description="Sort order"),
) -> PaginatedMoviesResponse:
    """Retrieve catalog movies with pagination and optional metadata filters."""
    query = select(Movie)

    if year is not None:
        query = query.where(Movie.year == year)
    if year_min is not None:
        query = query.where(Movie.year >= year_min)
    if year_max is not None:
        query = query.where(Movie.year <= year_max)

    # For genres: Movie.genres is stored as JSON array of strings
    if genre:
        g_clean = genre.strip()
        query = query.where(cast(Movie.genres, String).ilike(f"%{g_clean}%"))

    offset = (page - 1) * limit

    # Popularity order comes from the train split held in MLService, so rank in Python
    if sort == "popular" and ml_service.popular_mids:
        rank = {mid: i for i, mid in enumerate(ml_service.popular_mids)}
        matches = (
            (await db.execute(query.where(Movie.ml_movie_id.in_(rank.keys())))).scalars().all()
        )
        ranked = sorted(matches, key=lambda m: rank[m.ml_movie_id])
        total = len(ranked)
        return PaginatedMoviesResponse(
            items=[MovieResponse.model_validate(m) for m in ranked[offset : offset + limit]],
            total=total,
            page=page,
            limit=limit,
            pages=(total + limit - 1) // limit,
        )

    # Total count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    # Pagination
    paged_query = query.order_by(Movie.id.asc()).offset(offset).limit(limit)
    items = (await db.execute(paged_query)).scalars().all()

    pages = (total + limit - 1) // limit if limit > 0 else 1

    return PaginatedMoviesResponse(
        items=[MovieResponse.model_validate(m) for m in items],
        total=total,
        page=page,
        limit=limit,
        pages=pages,
    )


@router.get(
    "/movies/poster-proxy",
    summary="Proxy TMDB poster image to bypass ISP/adblock blocks and CORS",
)
async def poster_proxy(
    path: str = Query(..., description="TMDB poster path e.g. /arw2vcBveWOVZr6pxd9XTd1TdQa.jpg"),
) -> Response:
    """Proxy image from TMDB to client, bypassing local ISP/DNS blocks."""
    clean_path = path.lstrip("/")
    if ".." in clean_path or not clean_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid poster path")
    url = f"https://image.tmdb.org/t/p/w500/{clean_path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                return Response(
                    content=resp.content,
                    media_type=resp.headers.get("content-type", "image/jpeg"),
                    headers={"Cache-Control": "public, max-age=604800"},
                )
            raise HTTPException(
                status_code=resp.status_code, detail="Could not fetch image from TMDB"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch image from TMDB: {e}",
        ) from e


@router.get(
    "/movies/{id}",
    response_model=MovieDetailResponse,
    summary="Get single movie details by ID",
)
async def get_movie(
    id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MovieDetailResponse:
    """Retrieve full movie metadata along with aggregate rating statistics."""
    query = select(Movie).where(Movie.id == id)
    movie = (await db.execute(query)).scalar_one_or_none()
    if movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")

    # Rating statistics
    stats_query = select(
        func.avg(Rating.rating).label("avg_rating"),
        func.count(Rating.id).label("ratings_count"),
    ).where(Rating.movie_id == id)
    stats_res = (await db.execute(stats_query)).first()

    avg_rating = float(stats_res.avg_rating) if stats_res and stats_res.avg_rating else None
    count_rating = int(stats_res.ratings_count) if stats_res else 0

    return MovieDetailResponse(
        id=movie.id,
        ml_movie_id=movie.ml_movie_id,
        tmdb_id=movie.tmdb_id,
        title=movie.title,
        year=movie.year,
        genres=movie.genres,
        overview=movie.overview,
        poster_path=movie.poster_path,
        average_rating=round(avg_rating, 2) if avg_rating else None,
        ratings_count=count_rating,
    )


@router.get(
    "/movies/{id}/similar",
    response_model=list[SimilarMovieResponse],
    summary="Get similar movies",
)
async def get_similar_movies(
    id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    k: int = Query(10, ge=1, le=50, description="Number of similar movies to return"),
) -> list[SimilarMovieResponse]:
    """Retrieve content-based similar movies based on genres, tags, and release era."""
    target_movie = (await db.execute(select(Movie).where(Movie.id == id))).scalar_one_or_none()
    if target_movie is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found.")

    # Use MLService similar_movies
    sim_results = ml_service.find_similar_movies(target_movie.ml_movie_id, k=k)

    if not sim_results:
        # Fallback to genre overlap in DB
        fallback_query = select(Movie).where(Movie.id != id).limit(k)
        movies = (await db.execute(fallback_query)).scalars().all()
        return [
            SimilarMovieResponse(
                id=m.id,
                ml_movie_id=m.ml_movie_id,
                title=m.title,
                year=m.year,
                genres=m.genres,
                similarity=0.5,
                poster_path=m.poster_path,
            )
            for m in movies
        ]

    # Map ml_movie_ids back to DB movies
    sim_mids = [r["movieId"] for r in sim_results]
    db_movies = (
        (await db.execute(select(Movie).where(Movie.ml_movie_id.in_(sim_mids)))).scalars().all()
    )
    mid_to_db_movie = {m.ml_movie_id: m for m in db_movies}

    response_items: list[SimilarMovieResponse] = []
    for r in sim_results:
        mid = r["movieId"]
        db_m = mid_to_db_movie.get(mid)
        if db_m:
            response_items.append(
                SimilarMovieResponse(
                    id=db_m.id,
                    ml_movie_id=db_m.ml_movie_id,
                    title=db_m.title,
                    year=db_m.year,
                    genres=db_m.genres,
                    similarity=round(float(r.get("similarity_score", 0.0)), 4),
                    poster_path=db_m.poster_path,
                )
            )

    return response_items


@router.get(
    "/search",
    response_model=list[SearchResultResponse],
    summary="Semantic movie search",
)
async def semantic_search(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    db: Annotated[AsyncSession, Depends(get_db)] = None,
    k: int = Query(10, ge=1, le=50, description="Max search results"),
    genre: str | None = Query(None, description="Optional genre filter"),
    year_min: int | None = Query(None, description="Optional min release year"),
    year_max: int | None = Query(None, description="Optional max release year"),
) -> list[SearchResultResponse]:
    """
    Search movies by semantic natural language meaning using sentence embeddings and cosine similarity.
    Supports pgvector cosine distance when running on PostgreSQL, with fallback to in-memory index.
    """
    # 1. Try in-memory SemanticSearch engine from ml_service
    if ml_service.semantic_search is not None:
        filters: dict[str, Any] = {}
        if genre:
            filters["genre"] = genre
        if year_min is not None:
            filters["year_min"] = year_min
        if year_max is not None:
            filters["year_max"] = year_max

        search_hits = ml_service.semantic_search.search(
            query=q, k=k, filters=filters if filters else None
        )

        # Map to DB movies if possible
        if db is not None and search_hits:
            mids = [h["movieId"] for h in search_hits]
            db_movies = (
                (await db.execute(select(Movie).where(Movie.ml_movie_id.in_(mids)))).scalars().all()
            )
            mid_to_movie = {m.ml_movie_id: m for m in db_movies}

            res: list[SearchResultResponse] = []
            for hit in search_hits:
                db_m = mid_to_movie.get(hit["movieId"])
                res.append(
                    SearchResultResponse(
                        id=db_m.id if db_m else hit.get("item_idx", 0),
                        ml_movie_id=hit["movieId"],
                        title=hit["title"],
                        year=hit["year"],
                        genres=hit["genres"],
                        overview=hit.get("overview"),
                        poster_path=db_m.poster_path if db_m else None,
                        similarity=round(float(hit["similarity"]), 4),
                    )
                )
            return res

    # 2. Database fallback if MLService index not loaded
    if db is not None:
        query = select(Movie).where(Movie.title.ilike(f"%{q}%")).limit(k)
        movies = (await db.execute(query)).scalars().all()
        return [
            SearchResultResponse(
                id=m.id,
                ml_movie_id=m.ml_movie_id,
                title=m.title,
                year=m.year,
                genres=m.genres,
                overview=m.overview,
                poster_path=m.poster_path,
                similarity=0.8,
            )
            for m in movies
        ]

    return []
