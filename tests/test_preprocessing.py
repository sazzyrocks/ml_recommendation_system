"""
tests/test_preprocessing.py
---------------------------
Unit tests for data preprocessing and temporal train/val/test split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.preprocessing import (
    PROCESSED_DATA_DIR,
    clean_movies,
    create_index_mappings,
    temporal_split_per_user,
)


def test_clean_movies():
    sample_movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": [
                "Toy Story (1995)",
                "One False Move (1992) ",
                "Documentary Without Year",
                "Complex Title: Part II (2004)",
            ],
            "genres": [
                "Adventure|Animation|Children|Comedy",
                "(no genres listed)",
                None,
                "Action|Thriller",
            ],
        }
    )

    cleaned = clean_movies(sample_movies)

    assert cleaned.loc[0, "title"] == "Toy Story"
    assert cleaned.loc[0, "year"] == 1995
    assert cleaned.loc[0, "genres"] == ["Adventure", "Animation", "Children", "Comedy"]

    assert cleaned.loc[1, "title"] == "One False Move"
    assert cleaned.loc[1, "year"] == 1992
    assert cleaned.loc[1, "genres"] == []

    assert cleaned.loc[2, "title"] == "Documentary Without Year"
    assert pd.isna(cleaned.loc[2, "year"])
    assert cleaned.loc[2, "genres"] == []

    assert cleaned.loc[3, "title"] == "Complex Title: Part II"
    assert cleaned.loc[3, "year"] == 2004
    assert cleaned.loc[3, "genres"] == ["Action", "Thriller"]


def test_create_index_mappings():
    sample_ratings = pd.DataFrame(
        {
            "userId": [10, 50, 10, 20],
            "movieId": [101, 103, 102, 101],
        }
    )
    sample_movies = pd.DataFrame(
        {
            "movieId": [101, 102, 103, 999],
        }
    )

    u_to_idx, idx_to_u, i_to_idx, idx_to_i = create_index_mappings(sample_ratings, sample_movies)

    # Check contiguous 0-based indexing
    assert set(u_to_idx.values()) == {0, 1, 2}
    assert set(i_to_idx.values()) == {0, 1, 2, 3}

    # Check bidirectional consistency
    for u, idx in u_to_idx.items():
        assert idx_to_u[idx] == u
    for i, idx in i_to_idx.items():
        assert idx_to_i[idx] == i


def test_temporal_split_logic_and_edge_cases():
    # User 1: 10 ratings with timestamps 100..109
    # User 2: 4 ratings (fewer than 5) with timestamps 200..203 -> must stay entirely in train
    ratings = pd.DataFrame(
        {
            "userId": [1] * 10 + [2] * 4,
            "movieId": list(range(1, 11)) + list(range(1, 5)),
            "rating": [4.0] * 14,
            "timestamp": list(range(100, 110)) + list(range(200, 204)),
        }
    )

    train_df, val_df, test_df = temporal_split_per_user(ratings, test_ratio=0.10, val_ratio=0.10)

    # User 2 has < 5 ratings -> 0 in val/test, all 4 in train
    assert len(train_df[train_df["userId"] == 2]) == 4
    assert len(val_df[val_df["userId"] == 2]) == 0
    assert len(test_df[test_df["userId"] == 2]) == 0

    # User 1 has 10 ratings: 1 test (10%), 1 val (10%), 8 train (80%)
    u1_train = train_df[train_df["userId"] == 1]
    u1_val = val_df[val_df["userId"] == 1]
    u1_test = test_df[test_df["userId"] == 1]

    assert len(u1_train) == 8
    assert len(u1_val) == 1
    assert len(u1_test) == 1

    # Check chronological ordering: train timestamps < val timestamps <= test timestamps
    assert u1_train["timestamp"].max() <= u1_val["timestamp"].min()
    assert u1_val["timestamp"].max() <= u1_test["timestamp"].min()


def test_saved_splits_no_overlap_and_temporal_order():
    """Verify the saved production parquet splits in data/processed."""
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Processed parquet files not yet generated.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # 1. Total row count conservation
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == 100836

    # 2. No interaction overlap between splits
    train_pairs = set(zip(train_df["user_idx"], train_df["item_idx"], strict=False))
    val_pairs = set(zip(val_df["user_idx"], val_df["item_idx"], strict=False))
    test_pairs = set(zip(test_df["user_idx"], test_df["item_idx"], strict=False))

    assert len(train_pairs & val_pairs) == 0, "Overlap found between train and validation splits!"
    assert len(train_pairs & test_pairs) == 0, "Overlap found between train and test splits!"
    assert len(val_pairs & test_pairs) == 0, "Overlap found between validation and test splits!"

    # 3. Temporal consistency per user:
    # No user's test timestamps can be earlier than their train timestamps.
    train_max_time = train_df.groupby("user_idx")["timestamp"].max()
    val_min_time = val_df.groupby("user_idx")["timestamp"].min()
    val_max_time = val_df.groupby("user_idx")["timestamp"].max()
    test_min_time = test_df.groupby("user_idx")["timestamp"].min()

    # For every user with validation ratings, val_min >= train_max
    for user_idx in val_min_time.index:
        assert (
            val_min_time[user_idx] >= train_max_time[user_idx]
        ), f"User {user_idx} has val rating earlier than train rating!"

    # For every user with test ratings, test_min >= val_max and test_min >= train_max
    for user_idx in test_min_time.index:
        assert (
            test_min_time[user_idx] >= train_max_time[user_idx]
        ), f"User {user_idx} has test rating earlier than train rating!"
        if user_idx in val_max_time:
            assert (
                test_min_time[user_idx] >= val_max_time[user_idx]
            ), f"User {user_idx} has test rating earlier than val rating!"
