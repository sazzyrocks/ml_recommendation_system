"""
tests/test_collaborative_model.py
---------------------------------
Unit tests for collaborative filtering algorithms in ml/collaborative_model.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.collaborative_model import BiasedMF, ImplicitALS, ItemKNN


@pytest.fixture
def synthetic_train_df():
    # 4 users, 5 items
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "item_idx": [0, 1, 2, 1, 2, 2, 3, 3, 4],
            "rating": [5.0, 4.0, 4.5, 4.0, 5.0, 4.0, 5.0, 4.5, 4.0],
            "timestamp": list(range(100, 109)),
        }
    )


def test_item_knn(synthetic_train_df):
    model = ItemKNN(n_neighbors=2)
    model.fit(synthetic_train_df)

    assert model.W is not None
    assert model.W.shape == (5, 5)

    # Predict rating for user 0 on item 3
    pred = model.predict(0, 3)
    assert 0.5 <= pred <= 5.0

    # User 0 has seen 0, 1, 2. Recommendations must exclude them
    recs = model.recommend(user_idx=0, k=2, exclude_seen=True)
    rec_items = [it for it, _ in recs]
    assert 0 not in rec_items
    assert 1 not in rec_items
    assert 2 not in rec_items


def test_biased_mf(synthetic_train_df, tmp_path):
    model = BiasedMF(n_factors=8, lr=0.01, reg=0.02, n_epochs=10, seed=42)
    model.fit(synthetic_train_df)

    assert model.P is not None and model.P.shape == (4, 8)
    assert model.Q is not None and model.Q.shape == (5, 8)

    pred = model.predict(user_idx=0, item_idx=3)
    assert 0.5 <= pred <= 5.0

    recs = model.recommend(user_idx=0, k=2, exclude_seen=True)
    assert len(recs) == 2
    rec_items = [it for it, _ in recs]
    assert 0 not in rec_items
    assert 1 not in rec_items
    assert 2 not in rec_items

    # Test saving factor matrices
    u_path = tmp_path / "u.npy"
    i_path = tmp_path / "i.npy"
    model.save_factors(u_path, i_path)
    assert u_path.exists()
    assert i_path.exists()
    assert np.load(u_path).shape == (4, 8)


def test_implicit_als(synthetic_train_df, tmp_path):
    model = ImplicitALS(n_factors=8, regularization=0.05, iterations=5, seed=42)
    model.fit(synthetic_train_df)

    assert model.model is not None
    assert model.model.user_factors.shape[0] == 4
    assert model.model.item_factors.shape[0] == 5

    recs = model.recommend(user_idx=0, k=2, exclude_seen=True)
    assert len(recs) <= 2
    rec_items = [it for it, _ in recs]
    assert 0 not in rec_items
    assert 1 not in rec_items
    assert 2 not in rec_items

    u_path = tmp_path / "u_als.npy"
    i_path = tmp_path / "i_als.npy"
    model.save_factors(u_path, i_path)
    assert u_path.exists()
    assert i_path.exists()
