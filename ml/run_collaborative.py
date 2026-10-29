"""
ml/run_collaborative.py
-----------------------
Tunes ItemKNN, BiasedMF, and ImplicitALS on validation split,
logs every run to reports/tuning_log.csv, appends the best configs to
reports/results_val.csv, and saves factor matrices to artifacts/.
"""

from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.collaborative_model import BiasedMF, ImplicitALS, ItemKNN
from ml.evaluate import evaluate_model

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TUNING_LOG_FILE = REPORTS_DIR / "tuning_log.csv"
RESULTS_FILE = REPORTS_DIR / "results_val.csv"


def log_tuning_run(
    model_name: str,
    params: dict,
    metrics: dict,
    k: int = 10,
    csv_path: Path = TUNING_LOG_FILE,
) -> None:
    """Log an individual hyperparameter tuning run to reports/tuning_log.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = [
        "timestamp",
        "model",
        "hyperparameters",
        "precision@k",
        "recall@k",
        "map@k",
        "ndcg@k",
        "hit_rate@k",
        "catalog_coverage",
        "rmse",
        "mae",
        "avg_latency_ms",
    ]

    param_str = "; ".join(f"{k_p}={v_p}" for k_p, v_p in params.items())

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "hyperparameters": param_str,
        "precision@k": round(metrics.get(f"precision@{k}", 0.0), 4),
        "recall@k": round(metrics.get(f"recall@{k}", 0.0), 4),
        "map@k": round(metrics.get(f"map@{k}", 0.0), 4),
        "ndcg@k": round(metrics.get(f"ndcg@{k}", 0.0), 4),
        "hit_rate@k": round(metrics.get(f"hit_rate@{k}", 0.0), 4),
        "catalog_coverage": round(metrics.get("catalog_coverage", 0.0), 4),
        "rmse": round(metrics["rmse"], 4) if metrics.get("rmse") is not None else "",
        "mae": round(metrics["mae"], 4) if metrics.get("mae") is not None else "",
        "avg_latency_ms": round(metrics.get("avg_latency_ms", 0.0), 2),
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_result(
    model_name: str,
    metrics: dict,
    k: int = 10,
    notes: str = "",
    csv_path: Path = RESULTS_FILE,
) -> None:
    """Append a best-config benchmark record to reports/results_val.csv."""
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
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Tune ItemKNN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. Tuning ItemKNN...")
    print("=" * 60)
    knn_grid = [
        {"n_neighbors": 10, "min_similarity": 0.0},
        {"n_neighbors": 20, "min_similarity": 0.0},
        {"n_neighbors": 40, "min_similarity": 0.0},
        {"n_neighbors": 20, "min_similarity": 0.1},
    ]

    best_knn_model = None
    best_knn_score = -1.0
    best_knn_params = {}
    best_knn_metrics = {}

    for params in knn_grid:
        model = ItemKNN(**params)
        model.fit(train_df)
        eval_metrics = evaluate_model(
            model, val_df, train_df=train_df, k=K, total_catalog_size=catalog_size
        )
        log_tuning_run("ItemKNN", params, eval_metrics, k=K)
        score = eval_metrics[f"ndcg@{K}"]
        print(
            f"  ItemKNN {params} -> NDCG@{K}: {score:.4f}, Hit@{K}: {eval_metrics[f'hit_rate@{K}']:.4f}, RMSE: {eval_metrics['rmse']:.4f}"
        )

        if score > best_knn_score:
            best_knn_score = score
            best_knn_model = model
            best_knn_params = params
            best_knn_metrics = eval_metrics

    print(f"Best ItemKNN config: {best_knn_params} (NDCG@{K} = {best_knn_score:.4f})")
    log_result(
        "ItemKNN",
        best_knn_metrics,
        k=K,
        notes=f"Mean-centered cosine ({best_knn_params})",
    )
    joblib.dump(best_knn_model, ARTIFACTS_DIR / "item_knn_model.joblib")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Tune BiasedMF (Explicit Ratings)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Tuning BiasedMF (Explicit SGD)...")
    print("=" * 60)
    mf_grid = [
        {"n_factors": 16, "reg": 0.02, "lr": 0.005, "n_epochs": 15},
        {"n_factors": 16, "reg": 0.05, "lr": 0.005, "n_epochs": 15},
        {"n_factors": 32, "reg": 0.02, "lr": 0.005, "n_epochs": 15},
        {"n_factors": 32, "reg": 0.05, "lr": 0.005, "n_epochs": 15},
        {"n_factors": 32, "reg": 0.10, "lr": 0.005, "n_epochs": 15},
    ]

    best_mf_model = None
    best_mf_rmse = 999.0
    best_mf_params = {}
    best_mf_metrics = {}

    for params in mf_grid:
        model = BiasedMF(**params)
        model.fit(train_df)
        eval_metrics = evaluate_model(
            model, val_df, train_df=train_df, k=K, total_catalog_size=catalog_size
        )
        log_tuning_run("BiasedMF", params, eval_metrics, k=K)
        rmse_val = eval_metrics["rmse"]
        print(
            f"  BiasedMF {params} -> RMSE: {rmse_val:.4f}, NDCG@{K}: {eval_metrics[f'ndcg@{K}']:.4f}, Hit@{K}: {eval_metrics[f'hit_rate@{K}']:.4f}"
        )

        if rmse_val < best_mf_rmse:
            best_mf_rmse = rmse_val
            best_mf_model = model
            best_mf_params = params
            best_mf_metrics = eval_metrics

    print(f"Best BiasedMF config (by RMSE): {best_mf_params} (RMSE = {best_mf_rmse:.4f})")
    log_result(
        "BiasedMF",
        best_mf_metrics,
        k=K,
        notes=f"Explicit SGD ({best_mf_params})",
    )
    assert best_mf_model is not None
    best_mf_model.save_factors(
        ARTIFACTS_DIR / "biased_mf_user_factors.npy",
        ARTIFACTS_DIR / "biased_mf_item_factors.npy",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Tune ImplicitALS (Implicit Feedback)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Tuning ImplicitALS...")
    print("=" * 60)
    als_grid = [
        {"n_factors": 16, "regularization": 0.01, "iterations": 15},
        {"n_factors": 16, "regularization": 0.10, "iterations": 15},
        {"n_factors": 32, "regularization": 0.01, "iterations": 15},
        {"n_factors": 32, "regularization": 0.10, "iterations": 15},
        {"n_factors": 64, "regularization": 0.10, "iterations": 15},
    ]

    best_als_model = None
    best_als_score = -1.0
    best_als_params = {}
    best_als_metrics = {}

    for params in als_grid:
        model = ImplicitALS(**params)
        model.fit(train_df)
        eval_metrics = evaluate_model(
            model, val_df, train_df=train_df, k=K, total_catalog_size=catalog_size
        )
        log_tuning_run("ImplicitALS", params, eval_metrics, k=K)
        score = eval_metrics[f"ndcg@{K}"]
        print(
            f"  ImplicitALS {params} -> NDCG@{K}: {score:.4f}, Hit@{K}: {eval_metrics[f'hit_rate@{K}']:.4f}, Recall@{K}: {eval_metrics[f'recall@{K}']:.4f}"
        )

        if score > best_als_score:
            best_als_score = score
            best_als_model = model
            best_als_params = params
            best_als_metrics = eval_metrics

    print(f"Best ImplicitALS config: {best_als_params} (NDCG@{K} = {best_als_score:.4f})")
    log_result(
        "ImplicitALS",
        best_als_metrics,
        k=K,
        notes=f"ALS on ratings>=4 ({best_als_params})",
    )
    assert best_als_model is not None
    best_als_model.save_factors(
        ARTIFACTS_DIR / "implicit_als_user_factors.npy",
        ARTIFACTS_DIR / "implicit_als_item_factors.npy",
    )

    print("\n" + "=" * 105)
    print("BEST CONFIGS BENCHMARK SUMMARY (Validation Split, K=10)")
    print("=" * 105)
    print(
        f"{'Model':<20} {'Precision@10':<14} {'Recall@10':<12} {'NDCG@10':<11} {'MAP@10':<11} {'HitRate@10':<13} {'RMSE':<10} {'Coverage':<10}"
    )
    print("-" * 105)
    for name, m in [
        ("ItemKNN", best_knn_metrics),
        ("BiasedMF", best_mf_metrics),
        ("ImplicitALS", best_als_metrics),
    ]:
        rmse_str = f"{m['rmse']:.4f}" if m.get("rmse") is not None else "N/A"
        print(
            f"{name:<20} "
            f"{m[f'precision@{K}']:<14.4f} "
            f"{m[f'recall@{K}']:<12.4f} "
            f"{m[f'ndcg@{K}']:<11.4f} "
            f"{m[f'map@{K}']:<11.4f} "
            f"{m[f'hit_rate@{K}']:<13.4f} "
            f"{rmse_str:<10} "
            f"{m['catalog_coverage']:<10.4f}"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()
