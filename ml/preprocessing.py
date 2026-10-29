"""
ml/preprocessing.py
-------------------
Data cleaning, ID-to-index mapping, and per-user temporal train/val/test split
for MovieLens ml-latest-small in CineMatch AI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Config Constants ──────────────────────────────────────────────────────────
RELEVANCE_THRESHOLD: float = 4.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Clean movies metadata.

    - Extracts 4-digit release year from parentheses in title.
    - Strips release year from title to produce a clean title.
    - Splits pipe-separated genres into a list of strings.
    - Maps '(no genres listed)' or missing genres to an empty list [].

    Parameters
    ----------
    movies_df : pd.DataFrame
        DataFrame loaded from movies.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned movies DataFrame with ['movieId', 'clean_title', 'year', 'genres', 'raw_title'].
    """
    df = movies_df.copy()

    years: list[int | None] = []
    clean_titles: list[str] = []

    year_pattern = re.compile(r"\s*\((\d{4})\)\s*$")

    for raw_title in df["title"]:
        title_str = str(raw_title).strip()
        match = year_pattern.search(title_str)
        if match:
            year = int(match.group(1))
            clean_title = title_str[: match.start()].strip()
        else:
            year = None
            clean_title = title_str
        years.append(year)
        clean_titles.append(clean_title)

    # Genre processing: split on '|', empty list for '(no genres listed)'
    def parse_genres(g: Any) -> list[str]:
        if pd.isna(g) or not g or str(g).strip() == "(no genres listed)":
            return []
        return [part.strip() for part in str(g).split("|") if part.strip()]

    df["raw_title"] = df["title"]
    df["title"] = clean_titles
    df["clean_title"] = clean_titles
    df["year"] = pd.Series(years, dtype="Int64", index=df.index)
    df["genres"] = df["genres"].apply(parse_genres)

    return df


def create_index_mappings(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Map raw userId and movieId to contiguous 0-based integer indices.

    Items are mapped using all movies in movies_df to ensure full catalog coverage.
    Users are mapped using all unique userIds in ratings_df.

    Returns
    -------
    tuple of (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
    """
    unique_users = sorted(ratings_df["userId"].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}

    unique_items = sorted(movies_df["movieId"].unique())
    item_to_idx = {mid: idx for idx, mid in enumerate(unique_items)}
    idx_to_item = {idx: mid for mid, idx in item_to_idx.items()}

    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def temporal_split_per_user(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.10,
    val_ratio: float = 0.10,
    min_ratings_threshold: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split.

    For each user:
    - Sort user interactions chronologically by timestamp.
    - If user has fewer than `min_ratings_threshold` (5) ratings:
      all ratings stay entirely in train.
    - Otherwise:
      - Last 10% -> test
      - Previous 10% -> validation
      - Remaining ratings -> train
      (Ensuring at least 1 interaction in val and test if user has >= 5 ratings).

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Interaction dataframe with ['user_idx', 'item_idx', 'timestamp', ...].
    test_ratio : float, default=0.10
    val_ratio : float, default=0.10
    min_ratings_threshold : int, default=5

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    # Group by user_idx (or userId)
    user_col = "user_idx" if "user_idx" in ratings_df.columns else "userId"

    for _, group in ratings_df.groupby(user_col):
        # Sort chronologically, stable tie-breaker by original index
        sorted_group = group.sort_values(by=["timestamp"], kind="mergesort")
        indices = sorted_group.index.tolist()
        n = len(indices)

        if n < min_ratings_threshold:
            train_indices.extend(indices)
            continue

        n_test = max(1, int(np.round(n * test_ratio)))
        n_val = max(1, int(np.round(n * val_ratio)))

        # Ensure train gets at least 1 interaction
        if n - n_test - n_val < 1:
            n_test = 1
            n_val = 1

        split_train_end = n - n_val - n_test
        split_val_end = n - n_test

        train_indices.extend(indices[:split_train_end])
        val_indices.extend(indices[split_train_end:split_val_end])
        test_indices.extend(indices[split_val_end:])

    train_df = ratings_df.loc[train_indices].copy().reset_index(drop=True)
    val_df = ratings_df.loc[val_indices].copy().reset_index(drop=True)
    test_df = ratings_df.loc[test_indices].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full data preprocessing, indexing, and splitting pipeline.

    Saves:
    - data/processed/train.parquet
    - data/processed/val.parquet
    - data/processed/test.parquet
    - data/processed/movies_cleaned.parquet
    - data/processed/user_to_idx.json
    - data/processed/item_to_idx.json

    Returns
    -------
    tuple of (train_df, val_df, test_df, movies_cleaned)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    movies_raw = pd.read_csv(raw_dir / "movies.csv")
    ratings_raw = pd.read_csv(raw_dir / "ratings.csv")

    # 1. Clean movies
    movies_cleaned = clean_movies(movies_raw)

    # 2. Build index mappings
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = create_index_mappings(
        ratings_raw, movies_cleaned
    )

    # Assign indices
    movies_cleaned["item_idx"] = movies_cleaned["movieId"].map(item_to_idx)
    ratings_indexed = ratings_raw.copy()
    ratings_indexed["user_idx"] = ratings_indexed["userId"].map(user_to_idx)
    ratings_indexed["item_idx"] = ratings_indexed["movieId"].map(item_to_idx)
    ratings_indexed["is_relevant"] = ratings_indexed["rating"] >= RELEVANCE_THRESHOLD

    # 3. Temporal split per user
    train_df, val_df, test_df = temporal_split_per_user(ratings_indexed)

    # 4. Save to disk
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    movies_cleaned.to_parquet(output_dir / "movies_cleaned.parquet", index=False)

    # Save JSON mappings (convert int keys to str for JSON serialization)
    with open(output_dir / "user_to_idx.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in user_to_idx.items()}, f, indent=2)

    with open(output_dir / "item_to_idx.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item_to_idx.items()}, f, indent=2)

    print("[OK] Pipeline completed.")
    print(f"     Train : {len(train_df):,} rows ({len(train_df)/len(ratings_raw)*100:.2f}%)")
    print(f"     Val   : {len(val_df):,} rows ({len(val_df)/len(ratings_raw)*100:.2f}%)")
    print(f"     Test  : {len(test_df):,} rows ({len(test_df)/len(ratings_raw)*100:.2f}%)")
    print(f"     Users : {len(user_to_idx):,}")
    print(f"     Movies: {len(item_to_idx):,}")

    return train_df, val_df, test_df, movies_cleaned


if __name__ == "__main__":
    run_pipeline()
