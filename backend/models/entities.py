"""
backend/models/entities.py
--------------------------
SQLAlchemy 2.0 database models for CineMatch AI:
- User (authentication & profile)
- Movie (catalog metadata, TMDB info, 384-dim dense embedding with HNSW index)
- Rating (user rating between 0.5 and 5.0, unique per user+movie)
- Watchlist (user saved movies, unique per user+movie)
"""

from __future__ import annotations

from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class User(Base):
    """Registered application user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    ratings: Mapped[list[Rating]] = relationship(
        "Rating",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    watchlist: Mapped[list[Watchlist]] = relationship(
        "Watchlist",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


class Movie(Base):
    """Movie catalog item with metadata and 384-dimensional dense semantic embedding."""

    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ml_movie_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    tmdb_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    genres: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    poster_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    embedding = mapped_column(Vector(384), nullable=True)

    # Relationships
    ratings: Mapped[list[Rating]] = relationship(
        "Rating",
        back_populates="movie",
        cascade="all, delete-orphan",
    )
    watchlist_entries: Mapped[list[Watchlist]] = relationship(
        "Watchlist",
        back_populates="movie",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "idx_movies_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Movie id={self.id} ml_movie_id={self.ml_movie_id} title={self.title} year={self.year}>"


class Rating(Base):
    """User rating for a movie (0.5 to 5.0 stars)."""

    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    movie_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("movies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="ratings")
    movie: Mapped[Movie] = relationship("Movie", back_populates="ratings")

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_rating"),
        CheckConstraint("rating >= 0.5 AND rating <= 5.0", name="chk_rating_range"),
    )

    def __repr__(self) -> str:
        return f"<Rating user_id={self.user_id} movie_id={self.movie_id} rating={self.rating}>"


class Watchlist(Base):
    """User's bookmarked / saved movies."""

    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    movie_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("movies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="watchlist")
    movie: Mapped[Movie] = relationship("Movie", back_populates="watchlist_entries")

    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_user_movie_watchlist"),)

    def __repr__(self) -> str:
        return f"<Watchlist user_id={self.user_id} movie_id={self.movie_id}>"
