"""
tests/test_baselines.py
-----------------------
Unit tests for baseline recommenders in ml/baselines.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.baselines import (
    PopularityRecommender,
    RandomRecommender,
    WeightedRatingRecommender,
)


@pytest.fixture
def sample_train_df():
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "item_idx": [10, 20, 20, 30, 20],  # item 20 has 3 votes, item 10 has 1, item 30 has 1
            "rating": [4.0, 5.0, 3.0, 2.0, 5.0],
            "timestamp": [100, 101, 102, 103, 104],
        }
    )


def test_random_recommender(sample_train_df):
    model = RandomRecommender(seed=42, all_item_indices=[10, 20, 30, 40, 50])
    model.fit(sample_train_df)

    # User 0 has seen 10 and 20. Recommendations must exclude seen items.
    recs = model.recommend(user_idx=0, k=3, exclude_seen=True)
    rec_items = [it for it, _ in recs]

    assert len(recs) == 3
    assert 10 not in rec_items
    assert 20 not in rec_items
    assert all(it in [30, 40, 50] for it in rec_items)


def test_popularity_recommender(sample_train_df):
    model = PopularityRecommender()
    model.fit(sample_train_df)

    # Item 20 is most popular (3 ratings)
    # User 2 has seen item 20. Top recommendation for user 2 must not be item 20.
    recs_u2 = model.recommend(user_idx=2, k=2, exclude_seen=True)
    rec_items_u2 = [it for it, _ in recs_u2]

    assert 20 not in rec_items_u2

    # User 3 has seen nothing. Top recommendation should be item 20.
    recs_u3 = model.recommend(user_idx=3, k=1, exclude_seen=True)
    assert recs_u3[0][0] == 20


def test_weighted_rating_recommender(sample_train_df):
    model = WeightedRatingRecommender(m=5.0)
    model.fit(sample_train_df)

    # Check fitted C matches mean rating
    assert model.C == pytest.approx(sample_train_df["rating"].mean())

    recs = model.recommend(user_idx=0, k=2, exclude_seen=True)
    assert len(recs) <= 2
    rec_items = [it for it, _ in recs]
    assert 10 not in rec_items
    assert 20 not in rec_items
