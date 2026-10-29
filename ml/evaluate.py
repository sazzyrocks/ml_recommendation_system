"""
ml/evaluate.py
--------------
Evaluation metrics and model benchmarking framework for CineMatch AI.

Includes:
- Ranking metrics: Precision@K, Recall@K, MAP@K, NDCG@K, HitRate@K
- Catalog coverage
- Rating prediction metrics: RMSE, MAE
- Common evaluation harness: evaluate_model(model, split_df, ...)
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ml.base import Recommender
from ml.preprocessing import RELEVANCE_THRESHOLD


def precision_at_k(
    recommended: list[int],
    relevant: set[int] | list[int],
    k: int = 10,
) -> float:
    """Compute Precision@K.

    Precision@K = (Number of recommended items up to rank K that are relevant) / K
    """
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    rel_set = set(relevant)
    hits = len(set(rec_k) & rel_set)
    return hits / float(k)


def recall_at_k(
    recommended: list[int],
    relevant: set[int] | list[int],
    k: int = 10,
) -> float:
    """Compute Recall@K.

    Recall@K = (Number of recommended items up to rank K that are relevant) / (Total relevant items)
    """
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & rel_set)
    return hits / float(len(rel_set))


def hit_rate_at_k(
    recommended: list[int],
    relevant: set[int] | list[int],
    k: int = 10,
) -> float:
    """Compute HitRate@K (1.0 if at least one relevant item appears in top-K, else 0.0)."""
    rec_k = recommended[:k]
    rel_set = set(relevant)
    return 1.0 if len(set(rec_k) & rel_set) > 0 else 0.0


def map_at_k(
    recommended: list[int],
    relevant: set[int] | list[int],
    k: int = 10,
) -> float:
    """Compute Average Precision @ K for a single query / user.

    AP@K = sum_{j=1..K} (Precision@j * rel(j)) / min(K, total_relevant)
    """
    rel_set = set(relevant)
    if not rel_set or k <= 0:
        return 0.0

    score = 0.0
    hits = 0

    for rank, item in enumerate(recommended[:k], start=1):
        if item in rel_set:
            hits += 1
            score += hits / float(rank)

    normalizer = min(len(rel_set), k)
    return score / float(normalizer)


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int] | list[int],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K with binary relevance.

    DCG@K = sum_{j=1..K} (rel(j) / log2(j + 1))
    IDCG@K = sum_{j=1..min(K, total_relevant)} (1.0 / log2(j + 1))
    NDCG@K = DCG@K / IDCG@K
    """
    rel_set = set(relevant)
    if not rel_set or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in rel_set:
            dcg += 1.0 / np.log2(rank + 1.0)

    # Compute Ideal DCG (all relevant items ranked at top)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def catalog_coverage(
    all_recommendations: Iterable[list[int]] | set[int],
    total_catalog_size: int,
) -> float:
    """Compute catalog coverage (fraction of catalog appearing in any top-K list)."""
    if total_catalog_size <= 0:
        return 0.0

    if isinstance(all_recommendations, set):
        unique_recommended = all_recommendations
    else:
        unique_recommended = set()
        for rec_list in all_recommendations:
            unique_recommended.update(rec_list)

    return float(len(unique_recommended) / total_catalog_size)


def rmse(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> float:
    """Root Mean Squared Error."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if len(t) == 0:
        return 0.0
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mae(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> float:
    """Mean Absolute Error."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if len(t) == 0:
        return 0.0
    return float(np.mean(np.abs(t - p)))


def evaluate_model(
    model: Recommender,
    split_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    k: int = 10,
    total_catalog_size: int | None = None,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate any model implementing the Recommender interface on a given split.

    Excludes items already present in the user's training history from recommendations.

    Parameters
    ----------
    model : Recommender
        Fitted model implementing fit() and recommend(user_idx, k, exclude_seen).
    split_df : pd.DataFrame
        Evaluation split (e.g. val_df or test_df) containing columns:
        ['user_idx', 'item_idx', 'rating'].
    train_df : pd.DataFrame, optional
        Training split used to verify / reinforce exclusion of training interactions.
    k : int, default=10
        Cutoff rank K.
    total_catalog_size : int, optional
        Total number of items in catalog (for coverage calculation).
    relevance_threshold : float, default=4.0
        Threshold above which a rating is considered positive/relevant.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed metrics (precision@k, recall@k, map@k, ndcg@k,
        hit_rate@k, catalog_coverage, rmse, mae, avg_latency_ms, evaluated_users).
    """
    user_col = "user_idx"
    item_col = "item_idx"
    rating_col = "rating"

    # Map user to seen train items
    seen_by_user: dict[int, set[int]] = {}
    if train_df is not None and not train_df.empty:
        for u, grp in train_df.groupby(user_col):
            seen_by_user[u] = set(grp[item_col].tolist())

    # Map user to ground-truth relevant items in evaluation split
    eval_users = sorted(split_df[user_col].unique())

    precisions: list[float] = []
    recalls: list[float] = []
    maps: list[float] = []
    ndcgs: list[float] = []
    hit_rates: list[float] = []
    latencies: list[float] = []
    all_recommended_items: set[int] = set()

    for user_idx in eval_users:
        user_eval = split_df[split_df[user_col] == user_idx]
        relevant_items = set(user_eval[user_eval[rating_col] >= relevance_threshold][item_col])

        # Measure recommendation latency
        t0 = time.perf_counter()
        recs = model.recommend(user_idx=user_idx, k=k, exclude_seen=True)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        # Extract recommended item indices
        rec_items = [item for item, _ in recs]

        # Double safeguard: filter out items from training history
        if user_idx in seen_by_user:
            rec_items = [it for it in rec_items if it not in seen_by_user[user_idx]]

        rec_items = rec_items[:k]
        all_recommended_items.update(rec_items)

        # Only compute ranking metrics for users with at least 1 relevant ground truth item
        if relevant_items:
            precisions.append(precision_at_k(rec_items, relevant_items, k=k))
            recalls.append(recall_at_k(rec_items, relevant_items, k=k))
            maps.append(map_at_k(rec_items, relevant_items, k=k))
            ndcgs.append(ndcg_at_k(rec_items, relevant_items, k=k))
            hit_rates.append(hit_rate_at_k(rec_items, relevant_items, k=k))

    # Catalog size calculation
    if total_catalog_size is None or total_catalog_size <= 0:
        total_catalog_size = split_df[item_col].nunique()
        if train_df is not None:
            total_catalog_size = max(
                total_catalog_size,
                int(train_df[item_col].max()) + 1,
            )

    cov = catalog_coverage(all_recommended_items, total_catalog_size)

    # Optional rating prediction metrics (RMSE / MAE)
    model_rmse: float | None = None
    model_mae: float | None = None
    try:
        y_true = []
        y_pred = []
        for _, row in split_df.iterrows():
            pred = model.predict(int(row[user_col]), int(row[item_col]))
            y_true.append(float(row[rating_col]))
            y_pred.append(pred)
        model_rmse = rmse(y_true, y_pred)
        model_mae = mae(y_true, y_pred)
    except (NotImplementedError, AttributeError):
        pass

    avg_latency_ms = float(np.mean(latencies) * 1000.0) if latencies else 0.0

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit_rate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "catalog_coverage": cov,
        "rmse": model_rmse,
        "mae": model_mae,
        "avg_latency_ms": avg_latency_ms,
        "evaluated_users": len(eval_users),
        "evaluated_relevant_users": len(precisions),
    }
