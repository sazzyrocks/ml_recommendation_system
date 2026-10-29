"""
ml/content_model.py
-------------------
Content-based recommendation model for CineMatch AI.

Features:
- Multi-hot genres encoding
- TF-IDF vectorization over aggregated user tags per movie
- One-hot decade release year buckets
- Configurable component weights
- Sparse item-item cosine similarity calculation
- similar_movies(movie_id, k) for "More like this"
- User profile representation using rating-centered weighted averages
- Serialization with joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ml.base import Recommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def extract_year_bucket(year: Any) -> str:
    """Bucket release year into decade categories."""
    if pd.isna(year) or year is None:
        return "year_unknown"
    try:
        y = int(year)
    except (ValueError, TypeError):
        return "year_unknown"

    if y < 1950:
        return "pre_1950"
    elif y < 1960:
        return "1950s"
    elif y < 1970:
        return "1960s"
    elif y < 1980:
        return "1970s"
    elif y < 1990:
        return "1980s"
    elif y < 2000:
        return "1990s"
    elif y < 2010:
        return "2000s"
    else:
        return "2010s"


class ContentRecommender(Recommender):
    """Content-based recommender using genres, tags TF-IDF, and release year buckets."""

    def __init__(
        self,
        weight_genres: float = 1.0,
        weight_tags: float = 1.2,
        weight_year: float = 0.4,
        max_tag_features: int = 1000,
        min_tag_df: int = 2,
    ):
        self.weight_genres = float(weight_genres)
        self.weight_tags = float(weight_tags)
        self.weight_year = float(weight_year)
        self.max_tag_features = max_tag_features
        self.min_tag_df = min_tag_df

        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.item_matrix: sparse.csr_matrix | None = None
        self.movies_df: pd.DataFrame | None = None

        # Mappings
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}

        # User profiles and seen histories
        self.user_profiles: dict[int, np.ndarray] = {}
        self.seen_by_user: dict[int, set[int]] = {}

    def build_item_features(
        self,
        movies_df: pd.DataFrame,
        tags_df: pd.DataFrame | None = None,
    ) -> sparse.csr_matrix:
        """Construct the combined weighted L2-normalized sparse item feature matrix."""
        self.movies_df = movies_df.copy().reset_index(drop=True)
        N = len(self.movies_df)

        # 1. Establish index mappings
        if "item_idx" in self.movies_df.columns:
            self.movie_id_to_idx = dict(
                zip(self.movies_df["movieId"], self.movies_df["item_idx"], strict=False)
            )
            self.idx_to_movie_id = dict(
                zip(self.movies_df["item_idx"], self.movies_df["movieId"], strict=False)
            )
        else:
            self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movies_df["movieId"])}
            self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}
            self.movies_df["item_idx"] = [
                self.movie_id_to_idx[mid] for mid in self.movies_df["movieId"]
            ]

        # 2. Genres Multi-Hot Encoding
        all_genres = sorted({g for glist in self.movies_df["genres"] for g in glist})
        genre_to_col = {g: i for i, g in enumerate(all_genres)}
        G = len(all_genres)

        g_rows, g_cols, g_data = [], [], []
        for row_idx, glist in enumerate(self.movies_df["genres"]):
            for g in glist:
                if g in genre_to_col:
                    g_rows.append(row_idx)
                    g_cols.append(genre_to_col[g])
                    g_data.append(1.0)

        genre_matrix = sparse.csr_matrix((g_data, (g_rows, g_cols)), shape=(N, G), dtype=np.float32)

        # 3. Tags TF-IDF
        if tags_df is not None and not tags_df.empty:
            tag_agg = (
                tags_df.groupby("movieId")["tag"]
                .apply(lambda s: " ".join(str(x) for x in s))
                .to_dict()
            )
        else:
            tag_agg = {}

        movie_tag_texts = [tag_agg.get(mid, "") for mid in self.movies_df["movieId"]]
        if any(text.strip() for text in movie_tag_texts):
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words="english",
                    min_df=self.min_tag_df,
                    max_features=self.max_tag_features,
                    token_pattern=r"(?u)\b[\w-]+\b",
                )
                tag_matrix = self.tfidf_vectorizer.fit_transform(movie_tag_texts).astype(np.float32)
            except ValueError:
                self.tfidf_vectorizer = None
                tag_matrix = sparse.csr_matrix((N, 0), dtype=np.float32)
        else:
            self.tfidf_vectorizer = None
            tag_matrix = sparse.csr_matrix((N, 0), dtype=np.float32)

        # 4. Release Year Bucket Encoding
        bucket_categories = [
            "pre_1950",
            "1950s",
            "1960s",
            "1970s",
            "1980s",
            "1990s",
            "2000s",
            "2010s",
            "year_unknown",
        ]
        bucket_to_col = {cat: i for i, cat in enumerate(bucket_categories)}
        y_rows = list(range(N))
        y_cols = [bucket_to_col[extract_year_bucket(y)] for y in self.movies_df["year"]]
        year_matrix = sparse.csr_matrix(
            ([1.0] * N, (y_rows, y_cols)), shape=(N, len(bucket_categories)), dtype=np.float32
        )

        # 5. Normalize sub-matrices and apply configurable component weights
        norm_genre = normalize(genre_matrix, norm="l2") * self.weight_genres
        norm_tags = normalize(tag_matrix, norm="l2") * self.weight_tags
        norm_year = normalize(year_matrix, norm="l2") * self.weight_year

        combined = sparse.hstack([norm_genre, norm_tags, norm_year]).tocsr()
        self.item_matrix = normalize(combined, norm="l2")
        return self.item_matrix

    def fit(
        self,
        train_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None,
        tags_df: pd.DataFrame | None = None,
    ) -> ContentRecommender:
        """Fit content model: build item matrix and compute user profile vectors."""
        if movies_df is None:
            cleaned_path = PROCESSED_DATA_DIR / "movies_cleaned.parquet"
            if cleaned_path.exists():
                movies_df = pd.read_parquet(cleaned_path)
            else:
                unique_items = sorted(int(i) for i in train_df["item_idx"].unique())
                movies_df = pd.DataFrame(
                    {
                        "movieId": unique_items,
                        "item_idx": unique_items,
                        "title": [f"Movie {i}" for i in unique_items],
                        "clean_title": [f"Movie {i}" for i in unique_items],
                        "year": [2000 for _ in unique_items],
                        "genres": [["Action", "Drama"] for _ in unique_items],
                    }
                )
        if tags_df is None:
            tags_path = RAW_DATA_DIR / "tags.csv"
            tags_df = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame()

        if self.item_matrix is None or self.movies_df is None:
            self.build_item_features(movies_df, tags_df)

        item_col = "item_idx"
        user_col = "user_idx"
        rating_col = "rating"

        self.seen_by_user = {}
        self.user_profiles = {}

        # Build user profile as rating-weighted average (rating minus user mean)
        assert self.item_matrix is not None
        for u, grp in train_df.groupby(user_col):
            items = grp[item_col].to_numpy()
            ratings = grp[rating_col].to_numpy(dtype=np.float32)
            self.seen_by_user[u] = set(items.tolist())

            user_mean = float(np.mean(ratings))
            weights = ratings - user_mean

            # If all ratings are identical (weights sum to zero), use raw ratings
            if np.allclose(weights, 0.0) or np.linalg.norm(weights) == 0:
                weights = ratings

            sub_matrix = self.item_matrix[items]
            profile = sub_matrix.T.dot(weights)  # (D,) array

            norm_val = np.linalg.norm(profile)
            if norm_val > 0.0:
                profile = profile / norm_val

            self.user_profiles[u] = profile

        return self

    def similar_movies(
        self,
        movie_identifier: int,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find the top-K most similar movies for the 'More like this' section.

        Parameters
        ----------
        movie_identifier : int
            Either raw `movieId` or 0-based `item_idx`.
        k : int, default=10
            Number of similar movies to return.

        Returns
        -------
        list[dict[str, Any]]
            List of movie info dictionaries with similarity scores.
        """
        if self.item_matrix is None or self.movies_df is None:
            raise ValueError("Model must be fitted before querying similar movies.")

        # Resolve to item_idx
        if movie_identifier in self.movie_id_to_idx:
            item_idx = self.movie_id_to_idx[movie_identifier]
        elif 0 <= movie_identifier < self.item_matrix.shape[0]:
            item_idx = movie_identifier
        else:
            raise KeyError(f"Movie identifier {movie_identifier} not found in catalog.")

        # Compute cosine similarity against all movies
        target_vec = self.item_matrix[item_idx]
        sims = (self.item_matrix @ target_vec.T).toarray().ravel()

        # Exclude the movie itself
        sims[item_idx] = -1.0

        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            row = self.movies_df.iloc[idx]
            results.append(
                {
                    "movieId": int(row["movieId"]),
                    "item_idx": int(idx),
                    "title": str(row["clean_title"]),
                    "year": int(row["year"]) if pd.notna(row["year"]) else None,
                    "genres": list(row["genres"]),
                    "similarity_score": round(float(sims[idx]), 4),
                }
            )
        return results

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate personalized top-K recommendations for user_idx by cosine similarity."""
        if self.item_matrix is None:
            raise ValueError("Model must be fitted before calling recommend.")

        if user_idx not in self.user_profiles:
            return []

        profile = self.user_profiles[user_idx]
        scores = self.item_matrix.dot(profile)

        if exclude_seen:
            seen = self.seen_by_user.get(user_idx, set())
            if seen:
                scores[list(seen)] = -9999.0

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def save(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        """Save fitted artifacts to disk using joblib."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tfidf_vectorizer, artifacts_dir / "content_vectorizer.joblib")
        joblib.dump(self.item_matrix, artifacts_dir / "content_item_matrix.joblib")
        joblib.dump(self, artifacts_dir / "content_model.joblib")
        print(f"[OK] Saved content artifacts to {artifacts_dir}/")

    @classmethod
    def load(cls, artifacts_dir: Path = ARTIFACTS_DIR) -> ContentRecommender:
        """Load fitted content model from artifacts."""
        model_path = artifacts_dir / "content_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Artifact not found: {model_path}")
        return joblib.load(model_path)
