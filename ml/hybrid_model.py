"""
ml/hybrid_model.py
------------------
Hybrid movie recommendation engine for CineMatch AI.

Combines:
1. Implicit Matrix Factorization (ImplicitALS) — personalized latent factor representations
2. ItemKNN — neighborhood-based collaborative similarity
3. Content-based recommender — genres, tags TF-IDF, and release decade profiling
4. Popularity / Bayesian weighted baseline — global popularity anchor

Features:
- Multi-source candidate generation (union of top-N per source)
- Per-source min-max normalization and weighted score fusion
- Filter seen train items
- Explainable recommendation reasons ("Because you enjoyed X", "Popular in Sci-Fi", etc.)
- Cold-start handling:
  * 0 ratings: popularity + onboarding genre choices
  * 1-5 ratings: content profile + item neighbors
  * new unrated movie: content/semantic similarity
- Fold-in adaptive personalization:
  * Closed-form least-squares solve for instant user latent factor updates without retraining
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.base import Recommender
from ml.baselines import PopularityRecommender
from ml.collaborative_model import ItemKNN
from ml.content_model import ContentRecommender
from ml.evaluate import evaluate_model
from ml.ranking import fuse_candidate_scores, generate_recommendation_reason

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


class HybridRecommender(Recommender):
    """
    Production-grade hybrid recommender combining Implicit ALS, ItemKNN,
    Content-based filtering, and Popularity baselines with explainable reasons.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        candidate_n: int = 100,
        artifacts_dir: Path = ARTIFACTS_DIR,
    ):
        self.weights = weights or {
            "implicit_als": 0.80,
            "item_knn": 0.12,
            "content": 0.08,
            "popularity": 0.00,
        }
        self.candidate_n = candidate_n
        self.artifacts_dir = Path(artifacts_dir)

        # Sub-models and factor matrices
        self.implicit_als_user_factors: np.ndarray | None = None
        self.implicit_als_item_factors: np.ndarray | None = None
        self.item_knn: ItemKNN | None = None
        self.content_model: ContentRecommender | None = None
        self.popularity_model: PopularityRecommender | None = None

        # Precomputed catalog metadata
        self.movies_df: pd.DataFrame | None = None
        self.item_idx_to_title: dict[int, str] = {}
        self.item_idx_to_genres: dict[int, list[str]] = {}
        self.item_idx_to_mid: dict[int, int] = {}
        self.mid_to_item_idx: dict[int, int] = {}

        # User training history
        self.seen_by_user: dict[int, set[int]] = {}
        self.user_rated_items: dict[int, dict[int, float]] = {}

        # Cache Y^T Y for fast fold-in updates
        self._YtY: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> HybridRecommender:
        """
        Fit or load pre-trained component models and catalog mappings.
        """
        logger.info("Initializing HybridRecommender components...")

        # 1. Load movies catalog metadata
        movies_path = PROCESSED_DATA_DIR / "movies_cleaned.parquet"
        if movies_path.exists():
            self.movies_df = pd.read_parquet(movies_path)
            for _, r in self.movies_df.iterrows():
                i_idx = int(r["item_idx"])
                mid = int(r["movieId"])
                title = str(r.get("clean_title") or r.get("title") or "")
                g_val = r["genres"]
                genres = list(g_val) if isinstance(g_val, list | np.ndarray) else [str(g_val)]
                self.item_idx_to_title[i_idx] = title
                self.item_idx_to_genres[i_idx] = genres
                self.item_idx_to_mid[i_idx] = mid
                self.mid_to_item_idx[mid] = i_idx
        else:
            max_item = max(int(train_df["item_idx"].max()), 100) + 1
            unique_items = list(range(max_item))
            for i_idx in unique_items:
                self.item_idx_to_title[i_idx] = f"Movie {i_idx}"
                self.item_idx_to_genres[i_idx] = ["Action", "Drama"]
                self.item_idx_to_mid[i_idx] = i_idx
                self.mid_to_item_idx[i_idx] = i_idx
            self.movies_df = pd.DataFrame(
                {
                    "movieId": unique_items,
                    "item_idx": unique_items,
                    "title": [f"Movie {i}" for i in unique_items],
                    "clean_title": [f"Movie {i}" for i in unique_items],
                    "genres": [["Action", "Drama"] for _ in unique_items],
                    "year": [2020 for _ in unique_items],
                }
            )

        # 2. Build user histories from train_df
        self.seen_by_user = {}
        self.user_rated_items = {}
        for row in train_df.itertuples(index=False):
            u = int(row.user_idx)
            i = int(row.item_idx)
            r = float(row.rating)
            if u not in self.seen_by_user:
                self.seen_by_user[u] = set()
                self.user_rated_items[u] = {}
            self.seen_by_user[u].add(i)
            self.user_rated_items[u][i] = r

        # 3. Load or fit Popularity baseline
        self.popularity_model = PopularityRecommender()
        self.popularity_model.fit(train_df)

        # 4. Load or fit ItemKNN
        knn_path = self.artifacts_dir / "item_knn_model.joblib"
        if knn_path.exists():
            logger.info(f"Loading ItemKNN from {knn_path}...")
            self.item_knn = joblib.load(knn_path)
        else:
            logger.info("Fitting fresh ItemKNN model...")
            self.item_knn = ItemKNN(n_neighbors=10)
            self.item_knn.fit(train_df)

        # 5. Load or fit ContentRecommender
        content_path = self.artifacts_dir / "content_model.joblib"
        if content_path.exists():
            logger.info(f"Loading ContentRecommender from {content_path}...")
            self.content_model = joblib.load(content_path)
        else:
            logger.info("Fitting fresh ContentRecommender model...")
            self.content_model = ContentRecommender()
            self.content_model.fit(train_df, movies_df=self.movies_df)

        # 6. Load Implicit ALS factor matrices
        als_item_path = self.artifacts_dir / "implicit_als_item_factors.npy"
        als_user_path = self.artifacts_dir / "implicit_als_user_factors.npy"
        if als_item_path.exists() and als_user_path.exists():
            logger.info(f"Loading ImplicitALS factors from {self.artifacts_dir}...")
            self.implicit_als_item_factors = np.load(als_item_path)
            self.implicit_als_user_factors = np.load(als_user_path)
            # Precompute Y^T Y for fold-in
            Y = self.implicit_als_item_factors
            self._YtY = np.dot(Y.T, Y)
        else:
            logger.info("Artifacts missing; initializing fallback item factors for testing/CI...")
            n_items = max(8000, max(self.item_idx_to_mid.keys(), default=0) + 1)
            rng = np.random.default_rng(42)
            self.implicit_als_item_factors = rng.normal(0, 0.1, size=(n_items, 64)).astype(
                np.float32
            )
            self._YtY = np.dot(self.implicit_als_item_factors.T, self.implicit_als_item_factors)

        logger.info("HybridRecommender components ready.")
        return self

    def generate_candidates(
        self,
        user_idx: int,
        n: int = 100,
    ) -> dict[str, dict[int, float]]:
        """
        Generate candidate recommendations with raw scores from each available sub-model.
        """
        candidates: dict[str, dict[int, float]] = {}

        # 1. Implicit ALS candidates (dot product scoring)
        if (
            self.implicit_als_user_factors is not None
            and self.implicit_als_item_factors is not None
        ):
            if 0 <= user_idx < len(self.implicit_als_user_factors):
                u_factor = self.implicit_als_user_factors[user_idx]
                als_scores = np.dot(self.implicit_als_item_factors, u_factor)
                actual_n = min(n, len(als_scores))
                if actual_n > 0:
                    if actual_n < len(als_scores):
                        top_als_idx = np.argpartition(als_scores, -actual_n)[-actual_n:]
                    else:
                        top_als_idx = np.arange(len(als_scores))
                    top_als_idx = top_als_idx[np.argsort(-als_scores[top_als_idx])]
                    candidates["implicit_als"] = {int(i): float(als_scores[i]) for i in top_als_idx}

        # 2. ItemKNN candidates
        if self.item_knn is not None:
            knn_recs = self.item_knn.recommend(user_idx, k=n, exclude_seen=False)
            candidates["item_knn"] = {}
            for rank, entry in enumerate(knn_recs):
                if isinstance(entry, tuple | list):
                    candidates["item_knn"][int(entry[0])] = float(entry[1])
                else:
                    candidates["item_knn"][int(entry)] = float(n - rank)

        # 3. Content candidates
        if self.content_model is not None:
            content_recs = self.content_model.recommend(user_idx, k=n, exclude_seen=False)
            candidates["content"] = {}
            for rank, entry in enumerate(content_recs):
                if isinstance(entry, tuple | list):
                    candidates["content"][int(entry[0])] = float(entry[1])
                else:
                    candidates["content"][int(entry)] = float(n - rank)

        # 4. Popularity candidates
        if self.popularity_model is not None and self.popularity_model.ranked_items is not None:
            pop_items = self.popularity_model.ranked_items[:n]
            candidates["popularity"] = {
                int(item): float(n - rank) for rank, item in enumerate(pop_items)
            }

        return candidates

    def _find_best_seed_movie(self, candidate_idx: int, user_idx: int) -> str | None:
        """Find the movie in the user's history with highest similarity to candidate_idx."""
        user_history = self.seen_by_user.get(user_idx, set())
        if not user_history:
            return None

        # Check ItemKNN similarity matrix if available
        if self.item_knn is not None and self.item_knn.W is not None:
            sim_col = self.item_knn.W[:, candidate_idx].toarray().ravel()
            best_sim = -1.0
            best_seed = None
            for seed_idx in user_history:
                if 0 <= seed_idx < len(sim_col):
                    s = sim_col[seed_idx]
                    if s > best_sim:
                        best_sim = s
                        best_seed = seed_idx
            if best_seed is not None and best_sim > 0.05:
                return self.item_idx_to_title.get(best_seed)

        # Fallback to user's highest rated movie in history
        rated = self.user_rated_items.get(user_idx, {})
        if rated:
            top_seed = max(rated, key=rated.get)
            return self.item_idx_to_title.get(top_seed)

        return None

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
        return_reasons: bool = False,
    ) -> list[int] | list[dict[str, Any]]:
        """
        Produce top-K hybrid recommendations for a user.

        If return_reasons is False, returns list[int] (item indices) for evaluation harness compatibility.
        If return_reasons is True, returns list[dict] with title, genres, score, reason, and source breakdown.
        """
        # Check if user has no history (cold user)
        seen = self.seen_by_user.get(user_idx, set())
        if not seen and user_idx not in self.seen_by_user:
            return self.recommend_cold_user(k=k, return_reasons=return_reasons)

        # 1. Retrieve candidates
        candidates = self.generate_candidates(user_idx, n=self.candidate_n)

        # 2. Score fusion with min-max normalization
        fused_scores, breakdown = fuse_candidate_scores(candidates, self.weights)

        # 3. Filter seen items
        if exclude_seen:
            for seen_item in seen:
                fused_scores.pop(seen_item, None)

        if not fused_scores:
            # Fallback to popularity
            if self.popularity_model is not None:
                pop_recs = self.popularity_model.recommend(user_idx, k=k, exclude_seen=exclude_seen)
                if not return_reasons:
                    return pop_recs
                return [
                    {
                        "item_idx": item,
                        "title": self.item_idx_to_title.get(item, f"Movie #{item}"),
                        "score": 0.5,
                        "reason": "Trending across CineMatch viewers",
                        "top_source": "popularity",
                    }
                    for item in pop_recs
                ]
            return []

        # 4. Sort by fused score
        sorted_items = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:k]

        if not return_reasons:
            return [(int(item), float(fused_scores[item])) for item in sorted_items]

        # 5. Format results with reasons
        results: list[dict[str, Any]] = []
        for item in sorted_items:
            score = fused_scores[item]
            item_sources = breakdown.get(item, {})
            top_source = max(item_sources, key=item_sources.get) if item_sources else "popularity"

            # Identify seed movie if relevant
            seed_title = None
            if top_source in ("item_knn", "content"):
                seed_title = self._find_best_seed_movie(item, user_idx)

            genres = self.item_idx_to_genres.get(item, [])
            genre_hint = genres[0] if genres else None
            reason = generate_recommendation_reason(top_source, seed_title, genre_hint)

            results.append(
                {
                    "item_idx": item,
                    "movieId": self.item_idx_to_mid.get(item),
                    "title": self.item_idx_to_title.get(item, f"Movie #{item}"),
                    "genres": genres,
                    "score": round(score, 4),
                    "reason": reason,
                    "top_source": top_source,
                    "source_scores": {k: round(v, 4) for k, v in item_sources.items()},
                }
            )

        return results

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict hybrid compatibility score for a user-item pair."""
        score = 0.0
        # If ALS available
        if (
            self.implicit_als_user_factors is not None
            and self.implicit_als_item_factors is not None
            and 0 <= user_idx < len(self.implicit_als_user_factors)
            and 0 <= item_idx < len(self.implicit_als_item_factors)
        ):
            score += float(
                np.dot(
                    self.implicit_als_user_factors[user_idx],
                    self.implicit_als_item_factors[item_idx],
                )
            )
        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Cold-Start Strategies
    # ──────────────────────────────────────────────────────────────────────────

    def recommend_cold_user(
        self,
        genre_preferences: list[str] | None = None,
        k: int = 10,
        return_reasons: bool = True,
    ) -> list[int] | list[dict[str, Any]]:
        """
        Cold-start: Brand-new user with 0 ratings.
        Uses popularity + optional onboarding genre selections.
        """
        if self.movies_df is None or self.popularity_model is None:
            return []

        pop_ranks = {item: rank for rank, item in enumerate(self.popularity_model.ranked_items)}

        if not genre_preferences:
            # Global popularity
            top_items = self.popularity_model.ranked_items[:k]
            if not return_reasons:
                return [
                    (int(i), float(1.0 - (pop_ranks.get(i, 0) / len(pop_ranks)))) for i in top_items
                ]
            return [
                {
                    "item_idx": i,
                    "movieId": self.item_idx_to_mid.get(i),
                    "title": self.item_idx_to_title.get(i, f"Movie #{i}"),
                    "genres": self.item_idx_to_genres.get(i, []),
                    "score": round(1.0 - (pop_ranks.get(i, 0) / len(pop_ranks)), 4),
                    "reason": "Popular onboarding recommendation",
                    "top_source": "popularity",
                }
                for i in top_items
            ]

        # Filter by selected onboarding genres
        pref_genres = {g.strip().lower() for g in genre_preferences}
        matching_items: list[int] = []

        for item in self.popularity_model.ranked_items:
            item_genres = {g.lower() for g in self.item_idx_to_genres.get(item, [])}
            if item_genres & pref_genres:
                matching_items.append(item)
            if len(matching_items) >= k:
                break

        if not return_reasons:
            return [
                (int(i), float(1.0 - (pop_ranks.get(i, 0) / len(pop_ranks))))
                for i in matching_items
            ]

        results: list[dict[str, Any]] = []
        for i in matching_items:
            matched = [g for g in self.item_idx_to_genres.get(i, []) if g.lower() in pref_genres]
            g_str = matched[0] if matched else "your favorite genre"
            results.append(
                {
                    "item_idx": i,
                    "movieId": self.item_idx_to_mid.get(i),
                    "title": self.item_idx_to_title.get(i, f"Movie #{i}"),
                    "genres": self.item_idx_to_genres.get(i, []),
                    "score": round(1.0 - (pop_ranks.get(i, 0) / len(pop_ranks)), 4),
                    "reason": f"Top rated in {g_str}",
                    "top_source": "popularity",
                }
            )
        return results

    def recommend_sparse_user(
        self,
        user_ratings: dict[int, float],
        k: int = 10,
        return_reasons: bool = True,
    ) -> list[int] | list[dict[str, Any]]:
        """
        Cold-start: User with 1-5 ratings.
        Bypasses dense factor decomposition; relies on Content-based profile + ItemKNN neighbors.
        """
        if not user_ratings:
            return self.recommend_cold_user(k=k, return_reasons=return_reasons)

        rated_items = set(user_ratings.keys())
        mean_rating = np.mean(list(user_ratings.values()))

        # 1. ItemKNN neighborhood scoring from rated movies
        knn_scores: dict[int, float] = {}
        if self.item_knn is not None and self.item_knn.W is not None:
            for seed_item, r in user_ratings.items():
                if 0 <= seed_item < self.item_knn.n_items:
                    sim_col = self.item_knn.W[:, seed_item].toarray().ravel()
                    weight = max(0.0, r - mean_rating + 0.5)
                    for target_item, sim in enumerate(sim_col):
                        if sim > 0 and target_item not in rated_items:
                            knn_scores[target_item] = (
                                knn_scores.get(target_item, 0.0) + sim * weight
                            )

        # 2. Content-based profile similarity
        content_scores: dict[int, float] = {}
        if self.content_model is not None and self.content_model.item_matrix is not None:
            # Build mini profile
            prof = np.zeros(self.content_model.item_matrix.shape[1], dtype=np.float32)
            total_w = 0.0
            for seed_item, r in user_ratings.items():
                if 0 <= seed_item < self.content_model.item_matrix.shape[0]:
                    w = r - mean_rating
                    prof += self.content_model.item_matrix[seed_item].toarray().ravel() * w
                    total_w += abs(w)
            if total_w > 0:
                prof /= total_w
                norm = np.linalg.norm(prof)
                if norm > 0:
                    prof /= norm
                    sims = self.content_model.item_matrix.dot(prof)
                    actual_c = min(100, len(sims))
                    if actual_c > 0:
                        if actual_c < len(sims):
                            top_c = np.argpartition(sims, -actual_c)[-actual_c:]
                        else:
                            top_c = np.arange(len(sims))
                        content_scores = {
                            int(i): float(sims[i]) for i in top_c if i not in rated_items
                        }

        # 3. Fuse sparse scores
        fused, breakdown = fuse_candidate_scores(
            {"item_knn": knn_scores, "content": content_scores},
            {"item_knn": 0.65, "content": 0.35},
        )

        for r_item in rated_items:
            fused.pop(r_item, None)

        if not fused and self.popularity_model is not None:
            return self.recommend_cold_user(k=k, return_reasons=return_reasons)

        sorted_items = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:k]
        if not return_reasons:
            return [(int(item), float(fused[item])) for item in sorted_items]

        results: list[dict[str, Any]] = []
        best_seed_id = max(user_ratings, key=user_ratings.get)
        best_seed_title = self.item_idx_to_title.get(best_seed_id, "your rated movies")

        for item in sorted_items:
            score = fused[item]
            results.append(
                {
                    "item_idx": item,
                    "movieId": self.item_idx_to_mid.get(item),
                    "title": self.item_idx_to_title.get(item, f"Movie #{item}"),
                    "genres": self.item_idx_to_genres.get(item, []),
                    "score": round(score, 4),
                    "reason": f"Because you rated {best_seed_title}",
                    "top_source": "item_knn",
                }
            )
        return results

    def score_new_movie(
        self,
        user_idx: int,
        genres: list[str],
        tags: list[str] | None = None,
        movie_embedding: np.ndarray | None = None,
    ) -> float:
        """
        Score a brand-new unrated movie using content and semantic similarity against user profile.
        """
        if self.content_model is None or self.content_model.user_profiles is None:
            return 0.5

        u_prof = self.content_model.user_profiles.get(user_idx)
        if u_prof is None:
            return 0.5

        # Measure genre overlap with user's profile
        user_genres = []
        for seen_item in self.seen_by_user.get(user_idx, set()):
            user_genres.extend(self.item_idx_to_genres.get(seen_item, []))

        if user_genres:
            user_g_counts = pd.Series(user_genres).value_counts(normalize=True)
            overlap = sum(user_g_counts.get(g, 0.0) for g in genres)
            return float(np.clip(overlap, 0.0, 1.0))
        return 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Fold-in Adaptive Personalization
    # ──────────────────────────────────────────────────────────────────────────

    def fold_in_user(
        self,
        user_ratings: dict[int, float],
        alpha: float = 40.0,
        reg: float = 0.1,
    ) -> np.ndarray:
        r"""
        Compute an instant user latent vector via closed-form ridge regression / ALS step:
        x_u = (Y^T C_u Y + \lambda I)^{-1} Y^T C_u p_u

        Allows immediate updates when a user rates movies without retraining any models.
        """
        if self.implicit_als_item_factors is None:
            raise ValueError("Item factors not loaded.")

        Y = self.implicit_als_item_factors
        d = Y.shape[1]

        # Use precomputed Y^T Y
        if self._YtY is None:
            self._YtY = np.dot(Y.T, Y)

        A = self._YtY + reg * np.eye(d, dtype=np.float32)
        b = np.zeros(d, dtype=np.float32)

        for item_idx, r in user_ratings.items():
            if 0 <= item_idx < len(Y):
                y_i = Y[item_idx]
                # Implicit preference: rating >= 4.0 is positive
                p_i = 1.0 if r >= 4.0 else 0.0
                c_i = 1.0 + alpha * (r if r >= 4.0 else 0.5)

                # Rank-1 update to A: (c_i - 1) * y_i y_i^T
                A += (c_i - 1.0) * np.outer(y_i, y_i)
                # Update to b: c_i * p_i * y_i
                b += (c_i * p_i) * y_i

        # Solve linear system A * x_u = b
        x_u = np.linalg.solve(A, b)
        return x_u

    def recommend_fold_in_user(
        self,
        user_ratings: dict[int, float],
        k: int = 10,
        exclude_seen: bool = True,
        return_reasons: bool = True,
    ) -> list[int] | list[dict[str, Any]]:
        """
        Generate recommendations using folded-in user latent factors combined with
        content and ItemKNN neighborhoods.
        """
        x_u = self.fold_in_user(user_ratings)
        als_scores = np.dot(self.implicit_als_item_factors, x_u)

        actual_als = min(100, len(als_scores))
        if actual_als > 0:
            if actual_als < len(als_scores):
                top_als = np.argpartition(als_scores, -actual_als)[-actual_als:]
            else:
                top_als = np.arange(len(als_scores))
            cand_als = {int(i): float(als_scores[i]) for i in top_als}
        else:
            cand_als = {}

        # Sparse ItemKNN candidates
        rated_items = set(user_ratings.keys())
        cand_knn: dict[int, float] = {}
        if self.item_knn is not None and self.item_knn.W is not None:
            for seed_item, r in user_ratings.items():
                if 0 <= seed_item < self.item_knn.n_items and r >= 4.0:
                    sim_col = self.item_knn.W[:, seed_item].toarray().ravel()
                    for target, sim in enumerate(sim_col):
                        if sim > 0:
                            cand_knn[target] = cand_knn.get(target, 0.0) + sim

        # Fuse
        fused, breakdown = fuse_candidate_scores(
            {"implicit_als": cand_als, "item_knn": cand_knn},
            {"implicit_als": 0.70, "item_knn": 0.30},
        )

        if exclude_seen:
            for r_item in rated_items:
                fused.pop(r_item, None)

        sorted_items = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:k]
        if not return_reasons:
            return [(int(item), float(fused[item])) for item in sorted_items]

        results: list[dict[str, Any]] = []
        top_rated_item = max(user_ratings, key=user_ratings.get)
        seed_title = self.item_idx_to_title.get(top_rated_item, "your favorites")

        for item in sorted_items:
            score = fused[item]
            results.append(
                {
                    "item_idx": item,
                    "movieId": self.item_idx_to_mid.get(item),
                    "title": self.item_idx_to_title.get(item, f"Movie #{item}"),
                    "genres": self.item_idx_to_genres.get(item, []),
                    "score": round(score, 4),
                    "reason": f"Adaptive match based on your recent rating of {seed_title}",
                    "top_source": "implicit_als",
                }
            )
        return results


def tune_hybrid_weights(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    k: int = 10,
) -> dict[str, Any]:
    """
    Grid search over fusion weights on the validation split.
    Logs each configuration and returns the best performing weights.
    """
    logger.info("Starting hyperparameter tuning for HybridRecommender weights...")

    candidate_grids = [
        {"implicit_als": 0.80, "item_knn": 0.12, "content": 0.08, "popularity": 0.00},
        {"implicit_als": 0.85, "item_knn": 0.15, "content": 0.00, "popularity": 0.00},
        {"implicit_als": 0.75, "item_knn": 0.15, "content": 0.10, "popularity": 0.00},
        {"implicit_als": 0.70, "item_knn": 0.20, "content": 0.05, "popularity": 0.05},
        {"implicit_als": 0.60, "item_knn": 0.25, "content": 0.10, "popularity": 0.05},
    ]

    best_ndcg = -1.0
    best_config: dict[str, Any] = {}
    best_metrics: dict[str, float] = {}

    model = HybridRecommender()
    model.fit(train_df)

    tuning_records: list[dict[str, Any]] = []

    for idx, weights in enumerate(candidate_grids, 1):
        model.weights = weights
        logger.info(f"Evaluating Config {idx}/{len(candidate_grids)}: {weights}")
        metrics = evaluate_model(model, val_df, train_df=train_df, k=k)

        ndcg_val = float(metrics[f"ndcg@{k}"])
        recall_val = float(metrics[f"recall@{k}"])
        prec_val = float(metrics[f"precision@{k}"])
        hit_val = float(metrics[f"hit_rate@{k}"])

        rec = {
            "config_id": idx,
            "weights": json.dumps(weights),
            f"ndcg@{k}": ndcg_val,
            f"recall@{k}": recall_val,
            f"precision@{k}": prec_val,
            f"hit_rate@{k}": hit_val,
            "catalog_coverage": metrics["catalog_coverage"],
        }
        tuning_records.append(rec)
        logger.info(
            f"Config {idx} -> NDCG@{k}: {ndcg_val:.4f}, Recall@{k}: {recall_val:.4f}, HitRate@{k}: {hit_val:.4f}"
        )

        if ndcg_val > best_ndcg:
            best_ndcg = ndcg_val
            best_config = weights
            best_metrics = metrics

    logger.info(f"Best Config: {best_config} (NDCG@{k}: {best_ndcg:.4f})")
    return {
        "best_weights": best_config,
        "best_metrics": best_metrics,
        "tuning_history": tuning_records,
    }


if __name__ == "__main__":
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")

    # Run tuning
    res = tune_hybrid_weights(train_df, val_df, k=10)
    best_weights = res["best_weights"]
    best_metrics = res["best_metrics"]

    # Append to results_val.csv
    results_csv = REPORTS_DIR / "results_val.csv"
    ts = pd.Timestamp.now(tz="UTC").isoformat()
    new_row = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "model": "HybridRecommender",
                "k": 10,
                "precision@k": round(best_metrics["precision@10"], 4),
                "recall@k": round(best_metrics["recall@10"], 4),
                "map@k": round(best_metrics["map@10"], 4),
                "ndcg@k": round(best_metrics["ndcg@10"], 4),
                "hit_rate@k": round(best_metrics["hit_rate@10"], 4),
                "catalog_coverage": round(best_metrics["catalog_coverage"], 4),
                "avg_latency_ms": round(best_metrics["avg_latency_ms"], 2),
                "evaluated_users": int(best_metrics["evaluated_users"]),
                "notes": f"Fusion: ALS {best_weights['implicit_als']}, KNN {best_weights['item_knn']}, Content {best_weights['content']}, Pop {best_weights['popularity']}",
            }
        ]
    )

    new_row.to_csv(results_csv, mode="a", header=not results_csv.exists(), index=False)
    print("\nSuccessfully updated reports/results_val.csv with HybridRecommender!")
