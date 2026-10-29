"""
ml/baselines.py
---------------
Baseline recommendation algorithms for CineMatch AI:
1. RandomRecommender (sanity lower bound)
2. PopularityRecommender (ranked by rating count in train)
3. WeightedRatingRecommender (IMDB-style Bayesian average weighted rating)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ml.base import Recommender


class RandomRecommender(Recommender):
    """Sanity floor baseline.

    Recommends uniformly sampled unseen items from the catalog.
    """

    def __init__(self, seed: int = 42, all_item_indices: Sequence[int] | None = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.all_items: list[int] = list(all_item_indices) if all_item_indices is not None else []
        self.seen_by_user: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> RandomRecommender:
        """Record catalog items and user interaction histories."""
        item_col = "item_idx"
        user_col = "user_idx"

        if not self.all_items:
            self.all_items = sorted(train_df[item_col].unique().tolist())

        self.seen_by_user = train_df.groupby(user_col)[item_col].apply(set).to_dict()
        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        seen = self.seen_by_user.get(user_idx, set()) if exclude_seen else set()
        candidates = [it for it in self.all_items if it not in seen]

        if not candidates:
            return []

        sampled_size = min(k, len(candidates))
        # Random permutation sample
        sampled_indices = self.rng.choice(len(candidates), size=sampled_size, replace=False)
        sampled_items = [candidates[i] for i in sampled_indices]

        # Uniform dummy scores descending
        return [(item, float(sampled_size - rank)) for rank, item in enumerate(sampled_items)]


class PopularityRecommender(Recommender):
    """Popularity baseline.

    Ranks items by frequency of interaction (rating count v) in the training set.
    """

    def __init__(self):
        self.ranked_items: list[int] = []
        self.item_counts: dict[int, int] = {}
        self.seen_by_user: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> PopularityRecommender:
        """Compute rating count per item and sort descending."""
        item_col = "item_idx"
        user_col = "user_idx"

        counts = train_df.groupby(item_col).size().sort_values(ascending=False)
        self.ranked_items = counts.index.tolist()
        self.item_counts = counts.to_dict()

        self.seen_by_user = train_df.groupby(user_col)[item_col].apply(set).to_dict()
        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        seen = self.seen_by_user.get(user_idx, set()) if exclude_seen else set()

        recs: list[tuple[int, float]] = []
        for item in self.ranked_items:
            if item not in seen:
                recs.append((item, float(self.item_counts.get(item, 0))))
                if len(recs) == k:
                    break
        return recs


class WeightedRatingRecommender(Recommender):
    """Bayesian average weighted rating recommender (IMDB formula).

    Formula:
        W = (v / (v + m)) * R + (m / (v + m)) * C

    Where:
        v = number of ratings for the movie in train
        m = minimum votes weight / Bayesian prior weight (tuned on validation)
        R = average rating of the movie in train
        C = mean rating across the entire training dataset
    """

    def __init__(self, m: float = 25.0):
        self.m = float(m)
        self.C: float = 3.5
        self.ranked_items: list[int] = []
        self.item_scores: dict[int, float] = {}
        self.seen_by_user: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> WeightedRatingRecommender:
        """Compute item stats v and R, apply shrinkage toward C, and sort descending."""
        item_col = "item_idx"
        user_col = "user_idx"
        rating_col = "rating"

        self.C = float(train_df[rating_col].mean())

        stats = train_df.groupby(item_col).agg(
            v=(rating_col, "count"),
            R=(rating_col, "mean"),
        )

        v = stats["v"]
        R = stats["R"]
        m = self.m
        C = self.C

        stats["W"] = (v / (v + m)) * R + (m / (v + m)) * C
        sorted_stats = stats.sort_values(by="W", ascending=False)

        self.ranked_items = sorted_stats.index.tolist()
        self.item_scores = sorted_stats["W"].to_dict()

        self.seen_by_user = train_df.groupby(user_col)[item_col].apply(set).to_dict()
        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        seen = self.seen_by_user.get(user_idx, set()) if exclude_seen else set()

        recs: list[tuple[int, float]] = []
        for item in self.ranked_items:
            if item not in seen:
                recs.append((item, float(self.item_scores.get(item, self.C))))
                if len(recs) == k:
                    break
        return recs

    def tune_m(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidate_m: Sequence[float] = (1.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0),
        k: int = 10,
        metric: str = "ndcg@10",
    ) -> tuple[float, dict[float, float]]:
        """Tune hyperparameter m on the validation split.

        Returns
        -------
        best_m : float
            The m value that maximized the target validation metric.
        results : dict[float, float]
            Mapping of candidate m to metric value.
        """
        from ml.evaluate import evaluate_model

        best_m = self.m
        best_score = -1.0
        scores: dict[float, float] = {}

        for cand in candidate_m:
            model = WeightedRatingRecommender(m=cand).fit(train_df)
            eval_dict = evaluate_model(model, val_df, train_df=train_df, k=k)
            score = float(eval_dict.get(metric, 0.0))
            scores[cand] = score
            if score > best_score:
                best_score = score
                best_m = cand

        self.m = best_m
        self.fit(train_df)
        return best_m, scores
