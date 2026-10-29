"""
ml/run_baselines.py
-------------------
Evaluates baseline recommenders on the validation split (K=10),
tunes WeightedRatingRecommender m parameter, and logs results to reports/results_val.csv.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml.baselines import (
    PopularityRecommender,
    RandomRecommender,
    WeightedRatingRecommender,
)
from ml.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_FILE = REPORTS_DIR / "results_val.csv"


def log_result(
    model_name: str,
    metrics: dict,
    k: int = 10,
    notes: str = "",
    csv_path: Path = RESULTS_FILE,
) -> None:
    """Append a benchmark record to reports/results_val.csv with a timestamp."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = [
        "timestamp",
        "model",
        "k",
        "precision@k",
        "recall@k",
        "map@k",
        "ndcg@k",
        "hit_rate@k",
        "catalog_coverage",
        "avg_latency_ms",
        "evaluated_users",
        "notes",
    ]

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "k": k,
        "precision@k": round(metrics.get(f"precision@{k}", 0.0), 4),
        "recall@k": round(metrics.get(f"recall@{k}", 0.0), 4),
        "map@k": round(metrics.get(f"map@{k}", 0.0), 4),
        "ndcg@k": round(metrics.get(f"ndcg@{k}", 0.0), 4),
        "hit_rate@k": round(metrics.get(f"hit_rate@{k}", 0.0), 4),
        "catalog_coverage": round(metrics.get("catalog_coverage", 0.0), 4),
        "avg_latency_ms": round(metrics.get("avg_latency_ms", 0.0), 2),
        "evaluated_users": metrics.get("evaluated_users", 0),
        "notes": notes,
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    print("Loading data splits...")
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")
    movies_df = pd.read_parquet(PROCESSED_DATA_DIR / "movies_cleaned.parquet")
    catalog_size = len(movies_df)

    K = 10
    results_summary = []

    # 1. RandomRecommender
    print("\n--- 1. Evaluating RandomRecommender ---")
    random_model = RandomRecommender(seed=42, all_item_indices=movies_df["item_idx"].tolist())
    random_model.fit(train_df)
    random_metrics = evaluate_model(
        random_model, val_df, train_df=train_df, k=K, total_catalog_size=catalog_size
    )
    log_result("RandomRecommender", random_metrics, k=K, notes="Sanity floor (random sample)")
    results_summary.append(("Random", random_metrics))

    # 2. PopularityRecommender
    print("\n--- 2. Evaluating PopularityRecommender ---")
    pop_model = PopularityRecommender()
    pop_model.fit(train_df)
    pop_metrics = evaluate_model(
        pop_model, val_df, train_df=train_df, k=K, total_catalog_size=catalog_size
    )
    log_result("PopularityRecommender", pop_metrics, k=K, notes="Ranked by train rating count")
    results_summary.append(("Popularity", pop_metrics))

    # 3. WeightedRatingRecommender (tune m on val)
    print("\n--- 3. Tuning & Evaluating WeightedRatingRecommender ---")
    weighted_model = WeightedRatingRecommender()
    candidate_m = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0]
    best_m, tuning_scores = weighted_model.tune_m(
        train_df=train_df,
        val_df=val_df,
        candidate_m=candidate_m,
        k=K,
        metric=f"ndcg@{K}",
    )
    print(f"Validation tuning scores (NDCG@{K}):")
    for m_val, score in tuning_scores.items():
        print(f"  m = {m_val:>5.1f} -> NDCG@{K} = {score:.4f}")
    print(f"Selected best m: {best_m}")

    weighted_metrics = evaluate_model(
        weighted_model, val_df, train_df=train_df, k=K, total_catalog_size=catalog_size
    )
    log_result(
        "WeightedRatingRecommender",
        weighted_metrics,
        k=K,
        notes=f"Bayesian average (m={best_m} tuned on val)",
    )
    results_summary.append((f"WeightedRating (m={best_m})", weighted_metrics))

    # Print summary table
    print("\n" + "=" * 95)
    print(
        f"{'Model':<25} {'Precision@10':<14} {'Recall@10':<12} {'NDCG@10':<11} {'MAP@10':<11} {'HitRate@10':<13} {'Coverage':<10}"
    )
    print("-" * 95)
    for name, m in results_summary:
        print(
            f"{name:<25} "
            f"{m[f'precision@{K}']:<14.4f} "
            f"{m[f'recall@{K}']:<12.4f} "
            f"{m[f'ndcg@{K}']:<11.4f} "
            f"{m[f'map@{K}']:<11.4f} "
            f"{m[f'hit_rate@{K}']:<13.4f} "
            f"{m['catalog_coverage']:<10.4f}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
