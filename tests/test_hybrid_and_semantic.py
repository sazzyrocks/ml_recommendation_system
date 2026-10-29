"""
tests/test_hybrid_and_semantic.py
---------------------------------
Unit tests for semantic search, ranking/fusion utilities, and hybrid recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.fetch_tmdb import build_movie_documents
from ml.hybrid_model import HybridRecommender
from ml.ranking import (
    fuse_candidate_scores,
    generate_recommendation_reason,
    min_max_scale,
)
from ml.semantic_search import SemanticSearch


def test_min_max_scale_normal() -> None:
    scores = {1: 10.0, 2: 20.0, 3: 30.0}
    norm = min_max_scale(scores)
    assert norm[1] == pytest.approx(0.0)
    assert norm[2] == pytest.approx(0.5)
    assert norm[3] == pytest.approx(1.0)


def test_min_max_scale_constant() -> None:
    scores = {1: 5.0, 2: 5.0}
    norm = min_max_scale(scores)
    assert norm[1] == 1.0
    assert norm[2] == 1.0


def test_fuse_candidate_scores() -> None:
    sources = {
        "source_a": {101: 10.0, 102: 5.0},
        "source_b": {101: 2.0, 103: 8.0},
    }
    weights = {"source_a": 0.6, "source_b": 0.4}

    fused, breakdown = fuse_candidate_scores(sources, weights)
    assert 101 in fused
    assert 102 in fused
    assert 103 in fused
    # Source A max is 101 (scaled to 1.0), Source B max is 103 (scaled to 1.0)
    assert breakdown[101]["source_a"] == pytest.approx(0.6)
    assert breakdown[103]["source_b"] == pytest.approx(0.4)


def test_generate_recommendation_reason() -> None:
    r_knn = generate_recommendation_reason("item_knn", seed_movie_title="Toy Story")
    assert "Toy Story" in r_knn

    r_als = generate_recommendation_reason("implicit_als")
    assert "Personalized match" in r_als

    r_pop = generate_recommendation_reason("popularity", genre_hint="Sci-Fi")
    assert "Sci-Fi" in r_pop


def test_build_movie_documents() -> None:
    sample_movies = pd.DataFrame(
        [
            {
                "movieId": 1,
                "item_idx": 0,
                "title": "Toy Story",
                "clean_title": "Toy Story",
                "year": 1995,
                "genres": ["Animation", "Children"],
            },
            {
                "movieId": 2,
                "item_idx": 1,
                "title": "Jumanji",
                "clean_title": "Jumanji",
                "year": 1995,
                "genres": ["Adventure", "Fantasy"],
            },
        ]
    )
    sample_tags = pd.DataFrame(
        [
            {"movieId": 1, "tag": "pixar"},
            {"movieId": 1, "tag": "fun"},
        ]
    )

    docs_df = build_movie_documents(sample_movies, sample_tags)
    assert len(docs_df) == 2
    assert "document" in docs_df.columns
    doc_1 = docs_df.iloc[0]["document"]
    assert "Toy Story" in doc_1
    assert "1995" in doc_1
    assert "Animation" in doc_1


def test_semantic_search_with_filters() -> None:
    engine = SemanticSearch()
    engine.load_index()

    # Query with Comedy genre filter
    results = engine.search("heist robbery", k=5, filters={"genre": "Comedy"})
    assert len(results) <= 5
    assert len(results) > 0
    for r in results:
        assert any(g.lower() == "comedy" for g in r["genres"])
        assert "similarity" in r
        assert r["similarity"] > 0.0


def test_hybrid_recommender_recommendation() -> None:
    train_df = pd.DataFrame(
        [
            {"user_idx": 0, "item_idx": 10, "rating": 5.0},
            {"user_idx": 0, "item_idx": 20, "rating": 4.0},
            {"user_idx": 1, "item_idx": 10, "rating": 4.5},
            {"user_idx": 1, "item_idx": 30, "rating": 4.0},
        ]
    )
    recommender = HybridRecommender()
    recommender.fit(train_df)

    recs = recommender.recommend(user_idx=0, k=5, exclude_seen=True, return_reasons=False)
    assert isinstance(recs, list)
    for item, score in recs:
        assert item not in {10, 20}
        assert isinstance(score, float)

    recs_with_reasons = recommender.recommend(user_idx=0, k=5, return_reasons=True)
    assert len(recs_with_reasons) > 0
    assert "reason" in recs_with_reasons[0]
    assert "score" in recs_with_reasons[0]


def test_cold_start_scenarios() -> None:
    train_df = pd.DataFrame(
        [
            {"user_idx": 0, "item_idx": 10, "rating": 5.0},
            {"user_idx": 0, "item_idx": 20, "rating": 4.0},
        ]
    )
    h = HybridRecommender()
    h.fit(train_df)

    # 1. 0 ratings with genre preferences
    cold_recs = h.recommend_cold_user(genre_preferences=["Action"], k=3, return_reasons=True)
    assert len(cold_recs) <= 3
    assert all("Action" in r["genres"] for r in cold_recs)

    # 2. 1-5 ratings
    sparse_recs = h.recommend_sparse_user({0: 5.0, 1: 4.5}, k=3, return_reasons=True)
    assert len(sparse_recs) <= 3
    for r in sparse_recs:
        assert r["item_idx"] not in {0, 1}


def test_fold_in_personalization() -> None:
    train_df = pd.DataFrame(
        [
            {"user_idx": 0, "item_idx": 10, "rating": 5.0},
        ]
    )
    h = HybridRecommender()
    h.fit(train_df)

    # Compute fold-in vector for a new interaction
    user_factor = h.fold_in_user({7372: 5.0})
    assert isinstance(user_factor, np.ndarray)
    assert user_factor.shape == (64,)

    fold_recs = h.recommend_fold_in_user({7372: 5.0}, k=5, return_reasons=True)
    assert len(fold_recs) == 5
    for r in fold_recs:
        assert r["item_idx"] != 7372
        assert "reason" in r
