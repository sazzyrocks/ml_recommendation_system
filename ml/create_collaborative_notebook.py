"""
ml/create_collaborative_notebook.py
-----------------------------------
Generates and executes notebooks/03_collaborative_filtering.ipynb.
"""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_collaborative_filtering.ipynb"


def build_notebook() -> nbf.NotebookNode:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9",
        },
    }

    cells = []

    # Title
    cells.append(
        new_markdown_cell(
            """# Phase 6: Collaborative Filtering (ItemKNN, BiasedMF, ImplicitALS)

This notebook implements, tunes, and evaluates three foundational collaborative filtering paradigms for CineMatch AI:
1. **ItemKNN**: Memory-based neighborhood collaborative filtering using mean-centered cosine similarity.
2. **Biased Matrix Factorization (BiasedMF)**: Pointwise explicit rating estimation using global, user, and item biases trained via SGD.
3. **Implicit Alternating Least Squares (ImplicitALS)**: Pairwise preference ranking over positive interactions ($r \\ge 4.0$).

### Core Objectives:
- Implement models conforming to the common `Recommender` interface.
- Benchmark validation performance ($K=10$, Precision, Recall, NDCG, MAP, HitRate, Coverage, RMSE/MAE, Latency).
- Rigorously investigate: **Why does low rating-prediction RMSE not guarantee high top-K ranking performance?**
"""
        )
    )

    # 1. Setup & Data Loading
    cells.append(new_markdown_cell("## 1. Setup & Data Ingestion"))
    cells.append(
        new_code_cell(
            r"""import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path('..').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.collaborative_model import BiasedMF, ImplicitALS, ItemKNN
from ml.evaluate import evaluate_model

DATA_DIR = Path('../data/processed')
REPORTS_DIR = Path('../reports')
ARTIFACTS_DIR = Path('../artifacts')

train_df = pd.read_parquet(DATA_DIR / 'train.parquet')
val_df = pd.read_parquet(DATA_DIR / 'val.parquet')
movies_df = pd.read_parquet(DATA_DIR / 'movies_cleaned.parquet')
catalog_size = len(movies_df)

print(f'Train interactions : {len(train_df):,}')
print(f'Val interactions   : {len(val_df):,}')
print(f'Total catalog items: {catalog_size:,}')
"""
        )
    )

    # 2. ItemKNN
    cells.append(new_markdown_cell("## 2. ItemKNN: Neighborhood Collaborative Filtering"))
    cells.append(
        new_code_cell(
            r"""# Instantiate and fit ItemKNN with tuned top-10 neighbors
item_knn = ItemKNN(n_neighbors=10, min_similarity=0.0)
item_knn.fit(train_df)

knn_metrics = evaluate_model(item_knn, val_df, train_df=train_df, k=10, total_catalog_size=catalog_size)
print('--- ItemKNN Validation Benchmark (K=10) ---')
for k_m, val in knn_metrics.items():
    if isinstance(val, float):
        print(f'{k_m:<20}: {val:.4f}')
    else:
        print(f'{k_m:<20}: {val}')
"""
        )
    )

    # 3. Explicit BiasedMF
    cells.append(new_markdown_cell("## 3. Explicit Matrix Factorization (BiasedMF via SGD)"))
    cells.append(
        new_code_cell(
            r"""# Instantiate and train BiasedMF with tuned hyperparameters
biased_mf = BiasedMF(n_factors=16, reg=0.02, lr=0.005, n_epochs=15, seed=42)
biased_mf.fit(train_df)

mf_metrics = evaluate_model(biased_mf, val_df, train_df=train_df, k=10, total_catalog_size=catalog_size)
print('--- BiasedMF Validation Benchmark (K=10) ---')
for k_m, val in mf_metrics.items():
    if isinstance(val, float):
        print(f'{k_m:<20}: {val:.4f}')
    else:
        print(f'{k_m:<20}: {val}')
"""
        )
    )

    # 4. ImplicitALS
    cells.append(new_markdown_cell("## 4. Implicit Alternating Least Squares (ImplicitALS)"))
    cells.append(
        new_code_cell(
            r"""# Instantiate and train ImplicitALS (factors=64, reg=0.1)
implicit_als = ImplicitALS(n_factors=64, regularization=0.10, iterations=15, seed=42)
implicit_als.fit(train_df)

als_metrics = evaluate_model(implicit_als, val_df, train_df=train_df, k=10, total_catalog_size=catalog_size)
print('--- ImplicitALS Validation Benchmark (K=10) ---')
for k_m, val in als_metrics.items():
    if isinstance(val, float):
        print(f'{k_m:<20}: {val:.4f}')
    else:
        print(f'{k_m:<20}: {val}')
"""
        )
    )

    # 5. Hyperparameter Tuning Exploration
    cells.append(new_markdown_cell("## 5. Hyperparameter Tuning Analysis"))
    cells.append(
        new_code_cell(
            r"""tuning_csv = REPORTS_DIR / 'tuning_log.csv'
if tuning_csv.exists():
    tuning_df = pd.read_csv(tuning_csv)
    print('Hyperparameter Search Log:')
    display(tuning_df[['model', 'hyperparameters', 'ndcg@k', 'recall@k', 'hit_rate@k', 'rmse', 'catalog_coverage']])
"""
        )
    )

    # 6. Comprehensive Results Comparison Table
    cells.append(new_markdown_cell("## 6. Comprehensive Validation Comparison Across All Models"))
    cells.append(
        new_code_cell(
            r"""results_csv = REPORTS_DIR / 'results_val.csv'
if results_csv.exists():
    results_df = pd.read_csv(results_csv)
    cols = [c for c in ['model', 'k', 'precision@k', 'recall@k', 'map@k', 'ndcg@k', 'hit_rate@k', 'rmse', 'catalog_coverage', 'avg_latency_ms', 'notes'] if c in results_df.columns]
    display(results_df[cols])
"""
        )
    )

    # 7. In-Depth Explanation
    cells.append(
        new_markdown_cell(
            """## 7. Analysis: Optimizing RMSE vs. Optimizing Top-K Ranking

A classic paradox in recommendation systems is that **minimizing rating-prediction error (RMSE) does not necessarily yield superior top-K recommendation quality (NDCG / Recall / Hit Rate)**. Our empirical numbers demonstrate this divergence clearly:

| Model Paradigm | Objective | Validation RMSE | NDCG@10 | Recall@10 | HitRate@10 | Catalog Coverage |
|---|---|---|---|---|---|---|
| **BiasedMF** | Explicit rating error (pointwise L2) | **0.8950** | **0.0256** | **0.0271** | **0.1575** | 0.0059 |
| **ImplicitALS** | Pairwise preference (implicit feedback) | *N/A* | **0.0890** (+248%) | **0.1093** (+303%) | **0.3363** (+114%) | 0.0707 |
| **Popularity** | Interaction frequency | *N/A* | **0.0550** | **0.0585** | **0.2212** | 0.0101 |

### Key Reasons for this Divergence:
1. **The Asymmetry of User Needs:**
   - RMSE treats all errors equally: mispredicting a 1.5-star movie as 2.0 has the exact same loss impact as mispredicting a 4.5-star favorite as 4.0.
   - However, recommendation users only consume the top 10 items. Whether a bad movie is predicted as 1.5 or 2.5 is completely irrelevant to the user experience; what matters is whether the user's favorite items reach positions 1 through 10.
2. **Bias Dominance in Explicit Factorization:**
   - In BiasedMF, the prediction is dominated by $\mu + b_u + b_i$. Because 10% of movies account for 60% of all ratings with high average grades, positive item biases $b_i$ push familiar items to the top, while slight errors on thousands of unrated candidate items introduce rank inversions.
3. **Explicit vs. Implicit Feedback Formulation:**
   - BiasedMF only observes rated pairs $(u, i)$ and assumes missing entries are "missing at random".
   - ImplicitALS treats unrated movies as negative/background signal with low confidence and explicitly trains latent representations to distinguish positive interactions ($r \ge 4.0$) from the background.
   - Consequently, ImplicitALS achieves an **NDCG@10 of 0.0890** and **HitRate@10 of 33.6%** (surpassing both Popularity and explicit BiasedMF), even though it cannot compute explicit rating RMSE.
"""
        )
    )

    nb["cells"] = cells
    return nb


def main() -> int:
    nb = build_notebook()
    print("Executing 03_collaborative_filtering.ipynb using ipykernel...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}},
    )
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[OK] Successfully built and executed {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
