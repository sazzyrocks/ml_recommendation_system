"""
ml/run_content_model.py
-----------------------
Fits ContentRecommender, saves artifacts, evaluates on validation,
and logs metrics to reports/results_val.csv.
"""

from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.content_model import ContentRecommender
from ml.evaluate import evaluate_model

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_FILE = REPORTS_DIR / "results_val.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


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
    print("Loading data...")
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")
    movies_df = pd.read_parquet(PROCESSED_DATA_DIR / "movies_cleaned.parquet")
    tags_df = pd.read_csv(RAW_DATA_DIR / "tags.csv")
    catalog_size = len(movies_df)

    K = 10

    print("Fitting ContentRecommender (genres=1.0, tags=1.2, year=0.4)...")
    content_model = ContentRecommender(weight_genres=1.0, weight_tags=1.2, weight_year=0.4)
    content_model.fit(train_df=train_df, movies_df=movies_df, tags_df=tags_df)

    print("Saving artifacts to artifacts/...")
    content_model.save(ARTIFACTS_DIR)

    print(f"Evaluating ContentRecommender on validation (K={K})...")
    metrics = evaluate_model(
        content_model,
        val_df,
        train_df=train_df,
        k=K,
        total_catalog_size=catalog_size,
    )

    log_result(
        "ContentRecommender",
        metrics,
        k=K,
        notes="Genres (1.0) + Tags TF-IDF (1.2) + Year decade (0.4)",
    )

    print("\n" + "=" * 95)
    print(
        f"{'Model':<25} {'Precision@10':<14} {'Recall@10':<12} {'NDCG@10':<11} {'MAP@10':<11} {'HitRate@10':<13} {'Coverage':<10}"
    )
    print("-" * 95)
    print(
        f"{'ContentRecommender':<25} "
        f"{metrics[f'precision@{K}']:<14.4f} "
        f"{metrics[f'recall@{K}']:<12.4f} "
        f"{metrics[f'ndcg@{K}']:<11.4f} "
        f"{metrics[f'map@{K}']:<11.4f} "
        f"{metrics[f'hit_rate@{K}']:<13.4f} "
        f"{metrics['catalog_coverage']:<10.4f}"
    )
    print("=" * 95)

    # Similar movies inspection
    sample_movies = [
        (1, "Toy Story (1995)"),
        (2571, "The Matrix (1999)"),
        (79132, "Inception (2010)"),
        (296, "Pulp Fiction (1994)"),
        (593, "The Silence of the Lambs (1991)"),
    ]

    print("\n=== Movie-to-Movie Recommendations ('More Like This') ===")
    for mid, name in sample_movies:
        print(f"\nTarget: {name} [movieId={mid}]")
        sims = content_model.similar_movies(mid, k=5)
        for i, s in enumerate(sims, 1):
            genres_str = ", ".join(s["genres"])
            print(
                f"  {i}. {s['title']} ({s['year']}) - Sim: {s['similarity_score']:.3f} | Genres: [{genres_str}]"
            )


if __name__ == "__main__":
    main()
