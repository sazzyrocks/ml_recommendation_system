"""
ml/ranking.py
-------------
Ranking, score fusion, and explainability utilities for hybrid recommendation.
Provides min-max score normalization across candidate sources, weighted score fusion,
and human-interpretable recommendation reasons.
"""

from __future__ import annotations


def min_max_scale(scores: dict[int, float], eps: float = 1e-9) -> dict[int, float]:
    """
    Min-max normalize a dictionary of {item_id: score} to the range [0.0, 1.0].
    If all scores are identical or only one item exists, maps all scores to 1.0.
    """
    if not scores:
        return {}

    vals = list(scores.values())
    min_val = min(vals)
    max_val = max(vals)
    span = max_val - min_val

    if span < eps:
        return {k: 1.0 for k in scores}

    return {k: float((v - min_val) / (span + eps)) for k, v in scores.items()}


def fuse_candidate_scores(
    candidate_sources: dict[str, dict[int, float]],
    weights: dict[str, float],
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    """
    Combine candidates from multiple sources using per-source min-max normalization
    and weighted linear combination.

    Args:
        candidate_sources: Mapping of source_name -> {item_id: raw_score}.
        weights: Mapping of source_name -> weight.

    Returns:
        fused_scores: {item_id: total_fused_score}
        source_breakdown: {item_id: {source_name: weighted_normalized_score}}
    """
    fused_scores: dict[int, float] = {}
    source_breakdown: dict[int, dict[str, float]] = {}

    for source_name, raw_scores in candidate_sources.items():
        if not raw_scores:
            continue
        w = float(weights.get(source_name, 1.0))
        if w <= 0:
            continue

        norm_scores = min_max_scale(raw_scores)
        for item_idx, s_norm in norm_scores.items():
            weighted_s = w * s_norm
            fused_scores[item_idx] = fused_scores.get(item_idx, 0.0) + weighted_s

            if item_idx not in source_breakdown:
                source_breakdown[item_idx] = {}
            source_breakdown[item_idx][source_name] = weighted_s

    return fused_scores, source_breakdown


def generate_recommendation_reason(
    top_source: str,
    seed_movie_title: str | None = None,
    genre_hint: str | None = None,
) -> str:
    """
    Generate an intuitive, transparent reason explaining why a movie was recommended.
    """
    if top_source == "item_knn":
        if seed_movie_title:
            return f"Because you enjoyed {seed_movie_title}"
        return "Similar to movies in your watch history"

    elif top_source == "content":
        if seed_movie_title:
            return f"Shares genres and themes with {seed_movie_title}"
        if genre_hint:
            return f"Aligns with your interest in {genre_hint}"
        return "Matches your favorite genres and movie tags"

    elif top_source == "implicit_als":
        return "Personalized match for your overall viewing taste"

    elif top_source == "popularity":
        if genre_hint:
            return f"Popular among fans of {genre_hint}"
        return "Trending across CineMatch viewers"

    elif top_source == "semantic":
        return "Matches your narrative and stylistic search preferences"

    return "Recommended based on your movie preferences"
