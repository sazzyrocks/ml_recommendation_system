"""
tests/test_evaluate.py
----------------------
Unit tests for evaluation metrics using hand-computed ground truth examples.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml.base import Recommender
from ml.evaluate import (
    catalog_coverage,
    evaluate_model,
    hit_rate_at_k,
    mae,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


class DummyRecommender(Recommender):
    """Simple deterministic mock model for testing evaluate_model."""

    def __init__(self, rec_dict: dict[int, list[int]], rating_pred: float = 3.5):
        self.rec_dict = rec_dict
        self.rating_pred = rating_pred

    def fit(self, train_df: pd.DataFrame) -> DummyRecommender:
        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        items = self.rec_dict.get(user_idx, [])
        return [(item, 1.0 / (idx + 1)) for idx, item in enumerate(items[:k])]

    def predict(self, user_idx: int, item_idx: int) -> float:
        return self.rating_pred


def test_precision_at_k():
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 99}

    # At K=5: hits = {20, 40} -> 2 / 5 = 0.4
    assert precision_at_k(recommended, relevant, k=5) == 0.4

    # At K=3: rec[:3] = [10, 20, 30], hits = {20} -> 1 / 3
    assert math.isclose(precision_at_k(recommended, relevant, k=3), 1.0 / 3.0)

    # At K=1: rec[:1] = [10], 0 hits
    assert precision_at_k(recommended, relevant, k=1) == 0.0

    # No relevant items
    assert precision_at_k(recommended, set(), k=5) == 0.0


def test_recall_at_k():
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 99}  # total 3 relevant

    # At K=5: 2 hits out of 3 relevant
    assert math.isclose(recall_at_k(recommended, relevant, k=5), 2.0 / 3.0)

    # At K=3: 1 hit out of 3 relevant
    assert math.isclose(recall_at_k(recommended, relevant, k=3), 1.0 / 3.0)

    # 100% recall
    assert recall_at_k([20, 40, 99], relevant, k=3) == 1.0

    # Empty relevant set
    assert recall_at_k(recommended, set(), k=5) == 0.0


def test_hit_rate_at_k():
    recommended = [10, 20, 30]

    assert hit_rate_at_k(recommended, {20}, k=3) == 1.0
    assert hit_rate_at_k(recommended, {99}, k=3) == 0.0
    assert hit_rate_at_k(recommended, {20}, k=1) == 0.0  # 20 is at rank 2


def test_map_at_k():
    # recommended = [1, 2, 3, 4, 5], relevant = {1, 3} (size 2)
    # Rank 1: item 1 (hit 1), Prec@1 = 1/1 = 1.0
    # Rank 2: item 2 (miss)
    # Rank 3: item 3 (hit 2), Prec@3 = 2/3
    # AP@5 = (1.0 + 2/3) / min(2, 5) = (5/3) / 2 = 5/6 = 0.83333
    recommended = [1, 2, 3, 4, 5]
    relevant = {1, 3}
    expected_ap = (1.0 + (2.0 / 3.0)) / 2.0
    assert math.isclose(map_at_k(recommended, relevant, k=5), expected_ap)

    # Perfect ranking: [1, 2], relevant={1, 2} -> AP@2 = (1 + 1) / 2 = 1.0
    assert map_at_k([1, 2], {1, 2}, k=2) == 1.0

    # No hits
    assert map_at_k([4, 5], {1, 2}, k=2) == 0.0


def test_ndcg_at_k():
    # Perfect ranking of 2 items: DCG = IDCG -> NDCG = 1.0
    assert ndcg_at_k([1, 2, 3], {1, 2}, k=3) == 1.0

    # Hit at rank 2 instead of rank 1:
    # recommended = [2, 1, 3], relevant = {1}
    # DCG = 1 / log2(2 + 1) = 1 / log2(3)
    # IDCG = 1 / log2(1 + 1) = 1 / log2(2) = 1.0
    # NDCG = 1 / log2(3) = 0.6309297535714574
    recommended = [2, 1, 3]
    relevant = {1}
    expected_ndcg = (1.0 / np.log2(3.0)) / 1.0
    assert math.isclose(ndcg_at_k(recommended, relevant, k=3), expected_ndcg)

    # Zero hits
    assert ndcg_at_k([4, 5, 6], {1, 2}, k=3) == 0.0


def test_catalog_coverage():
    recs = [[1, 2], [2, 3], [3, 4]]
    # Unique items: {1, 2, 3, 4} (4 items)
    # Catalog size = 10 -> coverage = 4 / 10 = 0.4
    assert catalog_coverage(recs, total_catalog_size=10) == 0.4
    assert catalog_coverage(set(), total_catalog_size=10) == 0.0
    assert catalog_coverage(recs, total_catalog_size=0) == 0.0


def test_rmse_and_mae():
    y_true = [4.0, 3.0, 5.0]
    y_pred = [3.0, 3.0, 4.0]

    # Errors: [-1.0, 0.0, -1.0] -> squared: [1.0, 0.0, 1.0] -> mean: 2/3
    expected_rmse = math.sqrt(2.0 / 3.0)
    expected_mae = 2.0 / 3.0

    assert math.isclose(rmse(y_true, y_pred), expected_rmse)
    assert math.isclose(mae(y_true, y_pred), expected_mae)


def test_evaluate_model():
    # User 0 gets [1, 2, 3], User 1 gets [3, 4, 5]
    mock_model = DummyRecommender(
        rec_dict={
            0: [1, 2, 3],
            1: [3, 4, 5],
        },
        rating_pred=4.0,
    )

    split_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [1, 9, 4, 99],
            "rating": [5.0, 2.0, 4.5, 1.0],  # user 0 has rel item 1; user 1 has rel item 4
        }
    )

    train_df = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [100, 101],
            "rating": [5.0, 5.0],
        }
    )

    metrics = evaluate_model(
        model=mock_model,
        split_df=split_df,
        train_df=train_df,
        k=3,
        total_catalog_size=10,
        relevance_threshold=4.0,
    )

    assert "precision@3" in metrics
    assert "recall@3" in metrics
    assert "ndcg@3" in metrics
    assert "map@3" in metrics
    assert "hit_rate@3" in metrics
    assert "catalog_coverage" in metrics
    assert "avg_latency_ms" in metrics
    assert metrics["rmse"] is not None
    assert metrics["mae"] is not None

    # User 0: rec=[1, 2, 3], rel={1} -> Prec@3 = 1/3, Recall@3 = 1.0, HitRate@3 = 1.0
    # User 1: rec=[3, 4, 5], rel={4} -> Prec@3 = 1/3, Recall@3 = 1.0, HitRate@3 = 1.0
    assert math.isclose(metrics["precision@3"], 1.0 / 3.0)
    assert math.isclose(metrics["recall@3"], 1.0)
    assert math.isclose(metrics["hit_rate@3"], 1.0)

    # Unique recs: {1, 2, 3, 4, 5} (5 items) out of 10 -> coverage = 0.5
    assert math.isclose(metrics["catalog_coverage"], 0.5)
