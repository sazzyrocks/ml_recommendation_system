"""
ml/run_test_evaluation.py
-------------------------
Executes final evaluation experiments for CineMatch AI on the test split:
1. Runs all models once on test split and saves to reports/results_test.csv.
2. Performs hybrid ablation experiments (removing one source at a time).
3. Evaluates user segments: light users (<20 ratings) vs heavy users (>=20 ratings).
4. Measures popularity bias (average train rating count of recommended items).
5. Generates publication-ready figures in reports/figures/.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.baselines import PopularityRecommender, RandomRecommender, WeightedRatingRecommender
from ml.collaborative_model import BiasedMF, ImplicitALS, ItemKNN
from ml.content_model import ContentRecommender
from ml.evaluate import evaluate_model
from ml.hybrid_model import HybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_all_models_on_test() -> pd.DataFrame:
    """Run every model once on test.parquet and return results DataFrame."""
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")
    movies_df = pd.read_parquet(PROCESSED_DATA_DIR / "movies_cleaned.parquet")
    total_catalog = len(movies_df)

    logger.info(
        f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows, catalog: {total_catalog} items"
    )

    # Final chosen hyperparameters
    models: list[tuple[str, Any, str]] = [
        ("RandomRecommender", RandomRecommender(), "Sanity floor (random uniform)"),
        ("PopularityRecommender", PopularityRecommender(), "Ranked by train rating count"),
        (
            "WeightedRatingRecommender",
            WeightedRatingRecommender(m=100.0),
            "Bayesian average (m=100.0)",
        ),
        (
            "ContentRecommender",
            ContentRecommender(weight_genres=1.0, weight_tags=1.2, weight_year=0.4),
            "Genres + Tags TF-IDF + Decade buckets",
        ),
        ("ItemKNN", ItemKNN(n_neighbors=10), "Mean-centered cosine (k=10)"),
        (
            "BiasedMF",
            BiasedMF(n_factors=16, reg=0.02, lr=0.005, n_epochs=15),
            "Explicit SGD (d=16, reg=0.02, lr=0.005)",
        ),
        (
            "ImplicitALS",
            ImplicitALS(n_factors=64, regularization=0.1, iterations=15),
            "ALS on ratings>=4.0 (d=64, reg=0.1)",
        ),
        (
            "HybridRecommender",
            HybridRecommender(
                weights={"implicit_als": 0.85, "item_knn": 0.15, "content": 0.0, "popularity": 0.0}
            ),
            "Best tuned fusion (ALS 0.85, KNN 0.15)",
        ),
    ]

    results_records: list[dict[str, Any]] = []

    for name, model, notes in models:
        logger.info(f"--- Fitting & Evaluating {name} ---")
        t_start = time.perf_counter()
        model.fit(train_df)
        t_fit = time.perf_counter() - t_start
        logger.info(f"Fitted {name} in {t_fit:.2f}s. Running evaluation on test split...")

        metrics = evaluate_model(
            model=model,
            split_df=test_df,
            train_df=train_df,
            k=10,
            total_catalog_size=total_catalog,
        )

        record = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "model": name,
            "k": 10,
            "precision@k": round(metrics["precision@10"], 4),
            "recall@k": round(metrics["recall@10"], 4),
            "map@k": round(metrics["map@10"], 4),
            "ndcg@k": round(metrics["ndcg@10"], 4),
            "hit_rate@k": round(metrics["hit_rate@10"], 4),
            "catalog_coverage": round(metrics["catalog_coverage"], 4),
            "avg_latency_ms": round(metrics["avg_latency_ms"], 2),
            "evaluated_users": int(metrics["evaluated_users"]),
            "notes": notes,
        }
        results_records.append(record)
        logger.info(
            f"{name} -> Prec: {record['precision@k']}, Rec: {record['recall@k']}, "
            f"NDCG: {record['ndcg@k']}, Hit: {record['hit_rate@k']}, Cov: {record['catalog_coverage']}"
        )

    results_df = pd.DataFrame(results_records)
    results_csv = REPORTS_DIR / "results_test.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved test evaluation table to {results_csv}")
    return results_df


def run_hybrid_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Run ablation on HybridRecommender by removing one source at a time."""
    logger.info("--- Running Hybrid Ablation Study ---")
    configs = [
        (
            "Full 4-Way Hybrid",
            {"implicit_als": 0.70, "item_knn": 0.15, "content": 0.10, "popularity": 0.05},
        ),
        (
            "Ablation: No ALS",
            {"implicit_als": 0.00, "item_knn": 0.50, "content": 0.35, "popularity": 0.15},
        ),
        (
            "Ablation: No ItemKNN",
            {"implicit_als": 0.80, "item_knn": 0.00, "content": 0.15, "popularity": 0.05},
        ),
        (
            "Ablation: No Content",
            {"implicit_als": 0.75, "item_knn": 0.20, "content": 0.00, "popularity": 0.05},
        ),
        (
            "Ablation: No Popularity",
            {"implicit_als": 0.75, "item_knn": 0.15, "content": 0.10, "popularity": 0.00},
        ),
    ]

    model = HybridRecommender()
    model.fit(train_df)

    ablation_records: list[dict[str, Any]] = []
    base_ndcg = None

    for label, weights in configs:
        model.weights = weights
        metrics = evaluate_model(model, test_df, train_df=train_df, k=10)
        ndcg_val = metrics["ndcg@10"]
        if base_ndcg is None:
            base_ndcg = ndcg_val
            delta = 0.0
        else:
            delta = ndcg_val - base_ndcg

        ablation_records.append(
            {
                "variant": label,
                "weights": json.dumps(weights),
                "precision@10": round(metrics["precision@10"], 4),
                "recall@10": round(metrics["recall@10"], 4),
                "ndcg@10": round(ndcg_val, 4),
                "delta_ndcg@10": round(delta, 4),
                "hit_rate@10": round(metrics["hit_rate@10"], 4),
                "coverage": round(metrics["catalog_coverage"], 4),
            }
        )
        logger.info(
            f"{label} -> NDCG@10: {ndcg_val:.4f} (Delta: {delta:+.4f}), Recall@10: {metrics['recall@10']:.4f}"
        )

    ablation_df = pd.DataFrame(ablation_records)
    return ablation_df


def run_segment_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze model performance on light users (<20 train ratings) vs heavy users (>=20 train ratings)."""
    logger.info("--- Running User Segment Analysis ---")
    user_counts = train_df.groupby("user_idx").size()
    light_users = set(user_counts[user_counts < 20].index)
    heavy_users = set(user_counts[user_counts >= 20].index)

    light_test = test_df[test_df["user_idx"].isin(light_users)].copy()
    heavy_test = test_df[test_df["user_idx"].isin(heavy_users)].copy()

    logger.info(
        f"Light users in test: {light_test['user_idx'].nunique()}, Heavy users in test: {heavy_test['user_idx'].nunique()}"
    )

    eval_models = [
        ("Popularity", PopularityRecommender()),
        ("ItemKNN", ItemKNN(n_neighbors=10)),
        ("ImplicitALS", ImplicitALS(n_factors=64, regularization=0.1, iterations=15)),
        (
            "Hybrid",
            HybridRecommender(
                weights={"implicit_als": 0.85, "item_knn": 0.15, "content": 0.0, "popularity": 0.0}
            ),
        ),
    ]

    segment_records: list[dict[str, Any]] = []

    for name, model in eval_models:
        model.fit(train_df)

        m_light = evaluate_model(model, light_test, train_df=train_df, k=10)
        m_heavy = evaluate_model(model, heavy_test, train_df=train_df, k=10)

        segment_records.append(
            {
                "model": name,
                "segment": "Light (<20 ratings)",
                "n_users": int(m_light["evaluated_users"]),
                "ndcg@10": round(m_light["ndcg@10"], 4),
                "recall@10": round(m_light["recall@10"], 4),
                "hit_rate@10": round(m_light["hit_rate@10"], 4),
                "precision@10": round(m_light["precision@10"], 4),
            }
        )
        segment_records.append(
            {
                "model": name,
                "segment": "Heavy (>=20 ratings)",
                "n_users": int(m_heavy["evaluated_users"]),
                "ndcg@10": round(m_heavy["ndcg@10"], 4),
                "recall@10": round(m_heavy["recall@10"], 4),
                "hit_rate@10": round(m_heavy["hit_rate@10"], 4),
                "precision@10": round(m_heavy["precision@10"], 4),
            }
        )

    seg_df = pd.DataFrame(segment_records)
    return seg_df


def measure_popularity_bias(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Measure the average popularity (train rating count) of items recommended by each model."""
    logger.info("--- Measuring Popularity Bias of Recommendations ---")
    item_counts = train_df.groupby("item_idx").size().to_dict()
    eval_users = test_df["user_idx"].unique()

    eval_models = [
        ("Random", RandomRecommender()),
        ("Content-Based", ContentRecommender(weight_genres=1.0, weight_tags=1.2, weight_year=0.4)),
        ("ItemKNN", ItemKNN(n_neighbors=10)),
        ("ImplicitALS", ImplicitALS(n_factors=64, regularization=0.1, iterations=15)),
        (
            "Hybrid",
            HybridRecommender(
                weights={"implicit_als": 0.85, "item_knn": 0.15, "content": 0.0, "popularity": 0.0}
            ),
        ),
        ("Popularity", PopularityRecommender()),
    ]

    bias_records: list[dict[str, Any]] = []

    for name, model in eval_models:
        model.fit(train_df)
        all_pop_scores: list[float] = []

        for u in eval_users:
            recs = model.recommend(u, k=10, exclude_seen=True)
            for item_entry in recs:
                item_idx = item_entry[0] if isinstance(item_entry, tuple | list) else item_entry
                pop = item_counts.get(item_idx, 0)
                all_pop_scores.append(pop)

        avg_pop = float(np.mean(all_pop_scores)) if all_pop_scores else 0.0
        median_pop = float(np.median(all_pop_scores)) if all_pop_scores else 0.0
        p90_pop = float(np.percentile(all_pop_scores, 90)) if all_pop_scores else 0.0

        bias_records.append(
            {
                "model": name,
                "avg_item_popularity": round(avg_pop, 2),
                "median_item_popularity": round(median_pop, 2),
                "p90_item_popularity": round(p90_pop, 2),
            }
        )
        logger.info(f"{name} -> Mean Popularity: {avg_pop:.1f}, Median: {median_pop:.1f}")

    catalog_mean_pop = len(train_df) / 9742
    logger.info(f"Catalog baseline mean popularity: {catalog_mean_pop:.2f}")
    return pd.DataFrame(bias_records)


def generate_evaluation_figures(
    results_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    bias_df: pd.DataFrame,
) -> None:
    """Generate 4 publication-quality figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Figure 7: Test Model Comparison (NDCG@10 and Recall@10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_df = results_df.sort_values(by="ndcg@k", ascending=True).copy()

    colors = ["#4C72B0" if m != "HybridRecommender" else "#C44E52" for m in plot_df["model"]]
    bars1 = ax1.barh(plot_df["model"], plot_df["ndcg@k"], color=colors)
    ax1.set_title("Test NDCG@10 by Model", fontsize=13, fontweight="bold")
    ax1.set_xlabel("NDCG@10")
    for bar in bars1:
        ax1.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontsize=9,
        )

    bars2 = ax2.barh(plot_df["model"], plot_df["recall@k"], color=colors)
    ax2.set_title("Test Recall@10 by Model", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recall@10")
    for bar in bars2:
        ax2.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig7_path = FIGURES_DIR / "07_test_model_comparison.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig7_path}")

    # Figure 8: Hybrid Ablation (Delta NDCG@10)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    abl_sub = ablation_df[ablation_df["variant"] != "Full 4-Way Hybrid"].copy()
    colors_abl = ["#C44E52" if d < 0 else "#55A868" for d in abl_sub["delta_ndcg@10"]]
    bars_abl = ax.barh(abl_sub["variant"], abl_sub["delta_ndcg@10"], color=colors_abl)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(
        "Hybrid Component Ablation (Delta NDCG@10 vs Full Hybrid)", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Change in NDCG@10")
    for bar in bars_abl:
        w = bar.get_width()
        pos = w + (0.001 if w >= 0 else -0.005)
        ax.text(
            pos,
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig8_path = FIGURES_DIR / "08_hybrid_ablation.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig8_path}")

    # Figure 9: User Segment Analysis (Light vs Heavy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    palette_seg = {"Light (<20 ratings)": "#4C72B0", "Heavy (>=20 ratings)": "#E24A33"}

    sns.barplot(data=seg_df, x="model", y="ndcg@10", hue="segment", palette=palette_seg, ax=ax1)
    ax1.set_title("NDCG@10: Light vs Heavy Users", fontsize=13, fontweight="bold")
    ax1.set_ylabel("NDCG@10")
    ax1.set_xlabel("Model")
    ax1.legend(title="User Segment")

    sns.barplot(data=seg_df, x="model", y="recall@10", hue="segment", palette=palette_seg, ax=ax2)
    ax2.set_title("Recall@10: Light vs Heavy Users", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Recall@10")
    ax2.set_xlabel("Model")
    ax2.legend(title="User Segment")

    plt.tight_layout()
    fig9_path = FIGURES_DIR / "09_user_segment_performance.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig9_path}")

    # Figure 10: Popularity Bias by Model
    fig, ax = plt.subplots(figsize=(10, 5))
    bias_sorted = bias_df.sort_values(by="avg_item_popularity", ascending=True)
    bars_bias = ax.barh(bias_sorted["model"], bias_sorted["avg_item_popularity"], color="#4C72B0")
    catalog_mean = 80650 / 9742
    ax.axvline(
        catalog_mean,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Catalog Mean Activity ({catalog_mean:.1f} ratings/film)",
    )
    ax.set_title(
        "Popularity Bias: Average Train Ratings per Recommended Item",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Mean Ratings per Movie in Train History")
    ax.legend(loc="lower right")

    for bar in bars_bias:
        ax.text(
            bar.get_width() + 2.0,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig10_path = FIGURES_DIR / "10_popularity_bias_by_model.png"
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig10_path}")


def main() -> None:
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")

    # 1. Test Split Benchmark
    results_df = run_all_models_on_test()

    # 2. Hybrid Ablation
    ablation_df = run_hybrid_ablation(train_df, test_df)

    # 3. User Segment Analysis
    seg_df = run_segment_analysis(train_df, test_df)

    # 4. Popularity Bias
    bias_df = measure_popularity_bias(train_df, test_df)

    # 5. Generate Figures
    generate_evaluation_figures(results_df, ablation_df, seg_df, bias_df)

    logger.info("All final experiments completed successfully!")


if __name__ == "__main__":
    main()
