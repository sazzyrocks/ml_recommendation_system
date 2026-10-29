"""
backend/ml_service.py
---------------------
Singleton service managing trained recommender models and vector embedding search.
Loaded ONCE at FastAPI application startup via lifespan to ensure sub-millisecond
per-request inference without redundant disk or model loading overhead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml.content_model import ContentRecommender
from ml.hybrid_model import HybridRecommender
from ml.semantic_search import SemanticSearch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class MLService:
    """Manages loaded ML artifacts and provides fast in-memory recommendation inference."""

    def __init__(self) -> None:
        self.hybrid_model: HybridRecommender | None = None
        self.semantic_search: SemanticSearch | None = None
        self.content_model: ContentRecommender | None = None
        self.movies_df: pd.DataFrame | None = None

        # ID Mappings
        self.mid_to_item_idx: dict[int, int] = {}
        self.item_idx_to_mid: dict[int, int] = {}
        self.mid_to_db_id: dict[int, int] = {}

        # MovieLens movieIds ordered by train-split rating count (most popular first)
        self.popular_mids: list[int] = []

    def load_artifacts(self) -> None:
        """Load all precomputed artifacts, models, and embeddings into memory once."""
        logger.info("Initializing MLService artifacts...")

        # 1. Load movies metadata and mappings
        movies_path = PROCESSED_DATA_DIR / "movies_cleaned.parquet"
        if movies_path.exists():
            self.movies_df = pd.read_parquet(movies_path)
            for _, r in self.movies_df.iterrows():
                mid = int(r["movieId"])
                item_idx = int(r["item_idx"])
                self.mid_to_item_idx[mid] = item_idx
                self.item_idx_to_mid[item_idx] = mid

        # 2. Load Train split to initialize Hybrid Recommender
        train_path = PROCESSED_DATA_DIR / "train.parquet"
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
            counts = train_df["item_idx"].value_counts()
            self.popular_mids = [
                self.item_idx_to_mid[int(i)] for i in counts.index if int(i) in self.item_idx_to_mid
            ]
            logger.info("Fitting HybridRecommender from disk artifacts...")
            self.hybrid_model = HybridRecommender()
            self.hybrid_model.fit(train_df)
            self.content_model = self.hybrid_model.content_model

        # 3. Load Semantic Search engine
        emb_path = ARTIFACTS_DIR / "movie_embeddings.npy"
        if emb_path.exists():
            logger.info("Loading SemanticSearch engine...")
            self.semantic_search = SemanticSearch()
            self.semantic_search.load_index()

        logger.info("MLService artifacts fully loaded and ready for serving.")

    def recommend_for_user(
        self,
        user_ratings_db: dict[int, float],  # {ml_movie_id: rating}
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Generate personalized recommendations with reasons for a user using their DB ratings.
        Supports fold-in adaptive personalization for immediate updates.
        """
        if self.hybrid_model is None:
            return []

        if not user_ratings_db:
            # Cold-start user
            return self.hybrid_model.recommend_cold_user(k=k, return_reasons=True)

        # Map MovieLens movieId to contiguous item_idx
        sparse_item_ratings: dict[int, float] = {}
        for mid, r in user_ratings_db.items():
            if mid in self.mid_to_item_idx:
                sparse_item_ratings[self.mid_to_item_idx[mid]] = float(r)

        if not sparse_item_ratings:
            return self.hybrid_model.recommend_cold_user(k=k, return_reasons=True)

        if len(sparse_item_ratings) <= 5:
            # Sparse user strategy (Content + ItemKNN)
            return self.hybrid_model.recommend_sparse_user(
                sparse_item_ratings, k=k, return_reasons=True
            )

        # Dense user: instant closed-form fold-in least-squares
        return self.hybrid_model.recommend_fold_in_user(
            sparse_item_ratings, k=k, return_reasons=True
        )

    def find_similar_movies(self, ml_movie_id: int, k: int = 10) -> list[dict[str, Any]]:
        """Find movies similar to a given movie using content-based features."""
        if self.content_model is None:
            return []
        try:
            return self.content_model.similar_movies(ml_movie_id, k=k)
        except Exception as e:
            logger.warning(f"Error finding similar movies for movieId {ml_movie_id}: {e}")
            return []


# Global MLService instance
ml_service = MLService()
