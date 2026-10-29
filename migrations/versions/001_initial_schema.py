"""001_initial_schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-10-28 14:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Enable pgvector extension on Postgres
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    # 3. Movies table
    op.create_table(
        "movies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ml_movie_id", sa.Integer(), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("genres", sa.JSON(), nullable=False),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("poster_path", sa.String(length=500), nullable=True),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ml_movie_id"),
    )
    op.create_index(op.f("ix_movies_ml_movie_id"), "movies", ["ml_movie_id"], unique=True)
    op.create_index(op.f("ix_movies_tmdb_id"), "movies", ["tmdb_id"], unique=False)
    op.create_index(op.f("ix_movies_title"), "movies", ["title"], unique=False)
    op.create_index(op.f("ix_movies_year"), "movies", ["year"], unique=False)

    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_movies_embedding_hnsw 
            ON movies 
            USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 64);
            """
        )

    # 4. Ratings table
    op.create_table(
        "ratings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("rating >= 0.5 AND rating <= 5.0", name="chk_rating_range"),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "movie_id", name="uq_user_movie_rating"),
    )
    op.create_index(op.f("ix_ratings_movie_id"), "ratings", ["movie_id"], unique=False)
    op.create_index(op.f("ix_ratings_user_id"), "ratings", ["user_id"], unique=False)

    # 5. Watchlist table
    op.create_table(
        "watchlist",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "movie_id", name="uq_user_movie_watchlist"),
    )
    op.create_index(op.f("ix_watchlist_movie_id"), "watchlist", ["movie_id"], unique=False)
    op.create_index(op.f("ix_watchlist_user_id"), "watchlist", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_table("watchlist")
    op.drop_table("ratings")
    op.drop_table("movies")
    op.drop_table("users")
