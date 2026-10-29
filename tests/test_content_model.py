"""
tests/test_content_model.py
---------------------------
Unit tests for ContentRecommender in ml/content_model.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ml.content_model import ContentRecommender, extract_year_bucket


@pytest.fixture
def mini_dataset():
    movies_df = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "item_idx": [0, 1, 2, 3],
            "title": ["Toy Story", "Toy Story 2", "The Matrix", "The Matrix Reloaded"],
            "clean_title": ["Toy Story", "Toy Story 2", "The Matrix", "The Matrix Reloaded"],
            "year": [1995, 1999, 1999, 2003],
            "genres": [
                ["Animation", "Children", "Comedy"],
                ["Animation", "Children", "Comedy"],
                ["Action", "Sci-Fi"],
                ["Action", "Sci-Fi"],
            ],
        }
    )

    tags_df = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [1, 2, 3, 4],
            "tag": ["pixar fun", "pixar toys", "cyberpunk ai", "cyberpunk agents"],
            "timestamp": [100, 101, 102, 103],
        }
    )

    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [0, 1, 2],
            "rating": [5.0, 4.5, 5.0],
            "timestamp": [10, 11, 12],
        }
    )

    return movies_df, tags_df, train_df


def test_year_bucket():
    assert extract_year_bucket(1945) == "pre_1950"
    assert extract_year_bucket(1965) == "1960s"
    assert extract_year_bucket(1995) == "1990s"
    assert extract_year_bucket(2015) == "2010s"
    assert extract_year_bucket(None) == "year_unknown"
    assert extract_year_bucket("invalid") == "year_unknown"


def test_build_item_features(mini_dataset):
    movies_df, tags_df, _ = mini_dataset
    model = ContentRecommender(weight_genres=1.0, weight_tags=1.0, weight_year=0.5)
    item_mat = model.build_item_features(movies_df, tags_df)

    assert sparse.issparse(item_mat)
    assert item_mat.shape[0] == 4

    # Rows must be L2-normalized (norm approx 1.0)
    row_norms = np.sqrt(np.asarray(item_mat.power(2).sum(axis=1)).ravel())
    for norm in row_norms:
        assert norm == pytest.approx(1.0, abs=1e-4)


def test_similar_movies(mini_dataset):
    movies_df, tags_df, train_df = mini_dataset
    model = ContentRecommender()
    model.fit(train_df=train_df, movies_df=movies_df, tags_df=tags_df)

    # Toy Story (movieId 1) should be most similar to Toy Story 2 (movieId 2)
    recs = model.similar_movies(movie_identifier=1, k=2)
    assert len(recs) == 2
    assert recs[0]["movieId"] == 2
    assert recs[0]["similarity_score"] > recs[1]["similarity_score"]
    # Target movie itself must be excluded
    assert all(r["movieId"] != 1 for r in recs)


def test_user_recommendations_exclude_seen(mini_dataset):
    movies_df, tags_df, train_df = mini_dataset
    model = ContentRecommender()
    model.fit(train_df=train_df, movies_df=movies_df, tags_df=tags_df)

    # User 0 has seen items 0 (Toy Story) and 1 (Toy Story 2)
    recs = model.recommend(user_idx=0, k=2, exclude_seen=True)
    rec_items = [it for it, _ in recs]

    assert 0 not in rec_items
    assert 1 not in rec_items


def test_save_and_load(mini_dataset, tmp_path):
    movies_df, tags_df, train_df = mini_dataset
    model = ContentRecommender()
    model.fit(train_df=train_df, movies_df=movies_df, tags_df=tags_df)

    model.save(artifacts_dir=tmp_path)
    loaded = ContentRecommender.load(artifacts_dir=tmp_path)

    recs_orig = model.similar_movies(1, k=2)
    recs_loaded = loaded.similar_movies(1, k=2)
    assert recs_orig == recs_loaded
