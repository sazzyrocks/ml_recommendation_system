"""
backend/schemas.py
------------------
Pydantic schemas for request validation and response serialization across all API endpoints.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    detail: str


# ── Auth Schemas ──────────────────────────────────────────────────────────────


class UserRegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255, description="User email address")
    password: str = Field(
        ..., min_length=8, max_length=128, description="Plaintext password (min 8 chars)"
    )

    @field_validator("password")
    @classmethod
    def validate_password_rules(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one numerical digit.")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter.")
        return v


class UserLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Movie Schemas ─────────────────────────────────────────────────────────────


class MovieResponse(BaseModel):
    id: int
    ml_movie_id: int
    tmdb_id: int | None = None
    title: str
    year: int | None = None
    genres: list[str] = []
    overview: str | None = None
    poster_path: str | None = None

    model_config = ConfigDict(from_attributes=True)


class MovieDetailResponse(MovieResponse):
    average_rating: float | None = None
    ratings_count: int = 0


class PaginatedMoviesResponse(BaseModel):
    items: list[MovieResponse]
    total: int
    page: int
    limit: int
    pages: int


class SimilarMovieResponse(BaseModel):
    id: int
    ml_movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = []
    similarity: float
    poster_path: str | None = None


class SearchResultResponse(BaseModel):
    id: int
    ml_movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = []
    overview: str | None = None
    poster_path: str | None = None
    similarity: float


# ── Ratings Schemas ───────────────────────────────────────────────────────────


class RatingCreateRequest(BaseModel):
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0, description="Star rating between 0.5 and 5.0")


class RatingUpdateRequest(BaseModel):
    movie_id: int | None = None
    rating: float = Field(..., ge=0.5, le=5.0, description="Star rating between 0.5 and 5.0")


class RatingResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: float
    created_at: datetime
    movie: MovieResponse | None = None

    model_config = ConfigDict(from_attributes=True)


# ── Watchlist Schemas ─────────────────────────────────────────────────────────


class WatchlistCreateRequest(BaseModel):
    movie_id: int


class WatchlistResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    added_at: datetime
    movie: MovieResponse | None = None

    model_config = ConfigDict(from_attributes=True)


# ── Recommendation Schemas ────────────────────────────────────────────────────


class RecommendationItemResponse(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = []
    poster_path: str | None = None
    score: float
    reason: str
    top_source: str


class RecommendationsResponse(BaseModel):
    items: list[RecommendationItemResponse]
    user_id: int
    count: int
