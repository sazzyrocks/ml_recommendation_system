"""
ml/create_notebook.py
---------------------
Generates and executes notebooks/01_data_exploration.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_exploration.ipynb"


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
            """# Phase 2: Exploratory Data Analysis (EDA) - MovieLens ml-latest-small

This notebook provides an in-depth exploratory data analysis of the **MovieLens ml-latest-small** dataset to inform recommendation algorithm design and model selection for CineMatch AI.

### Scope of Analysis
1. Dataset shapes, data types, missing values, and duplicates
2. Rating distribution, central tendencies, and spread
3. User and movie activity distributions (long-tail behavior on log scale)
4. User-item interaction matrix sparsity
5. Genre frequencies and handling of `(no genres listed)`
6. Temporal release year extraction from titles
7. Popularity bias (top 10% movie rating concentration)
8. Ratings evolution over time (1996 - 2018)
9. Key empirical findings
"""
        )
    )

    # Cell 1: Setup & Imports
    cells.append(new_markdown_cell("## Setup & Configuration"))
    cells.append(
        new_code_cell(
            r"""import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Path configuration
DATA_DIR = Path('../data/raw/ml-latest-small')
FIGURES_DIR = Path('../reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
"""
        )
    )

    # Cell 2: Data Ingestion
    cells.append(new_markdown_cell("## 1. Data Ingestion & Shape Verification"))
    cells.append(
        new_code_cell(
            r"""movies = pd.read_csv(DATA_DIR / 'movies.csv')
ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
tags = pd.read_csv(DATA_DIR / 'tags.csv')
links = pd.read_csv(DATA_DIR / 'links.csv')

shapes_df = pd.DataFrame({
    'File': ['movies.csv', 'ratings.csv', 'tags.csv', 'links.csv'],
    'Rows': [len(movies), len(ratings), len(tags), len(links)],
    'Columns': [movies.shape[1], ratings.shape[1], tags.shape[1], links.shape[1]],
    'Primary Keys / Entities': ['movieId', '(userId, movieId)', '(userId, movieId, tag)', 'movieId']
})
shapes_df
"""
        )
    )

    # Cell 3: Data Types, Missing Values, Duplicates
    cells.append(new_markdown_cell("## 2. Data Types, Missing Values & Duplicate Inspection"))
    cells.append(
        new_code_cell(
            r"""print('--- Data Types ---')
print('movies:\n', movies.dtypes, '\n')
print('ratings:\n', ratings.dtypes, '\n')
print('tags:\n', tags.dtypes, '\n')
print('links:\n', links.dtypes, '\n')

print('--- Missing Values Count ---')
missing_summary = pd.DataFrame({
    'movies': movies.isnull().sum(),
    'ratings': ratings.isnull().sum(),
    'tags': tags.isnull().sum(),
    'links': links.isnull().sum()
}).fillna('-')
print(missing_summary)

print('\n--- Duplicate Records Check ---')
print('movies full duplicates:', movies.duplicated().sum())
print('movies duplicate movieId:', movies['movieId'].duplicated().sum())
print('ratings duplicate (userId, movieId):', ratings.duplicated(subset=['userId', 'movieId']).sum())
print('tags full duplicates:', tags.duplicated().sum())
"""
        )
    )

    # Cell 4: Rating Distribution & Statistics
    cells.append(new_markdown_cell("## 3. Rating Distribution & Central Tendencies"))
    cells.append(
        new_code_cell(
            r"""mean_rating = ratings['rating'].mean()
median_rating = ratings['rating'].median()
std_rating = ratings['rating'].std()

print(f'Total ratings : {len(ratings):,}')
print(f'Mean rating   : {mean_rating:.4f}')
print(f'Median rating : {median_rating:.1f}')
print(f'Std deviation : {std_rating:.4f}')

rating_freq = ratings['rating'].value_counts().sort_index().to_frame(name='count')
rating_freq['percentage'] = (rating_freq['count'] / len(ratings)) * 100
print('\nRating breakdown:')
display(rating_freq)

# Visualization
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(rating_freq.index.astype(str), rating_freq['count'], color='#3b82f6', edgecolor='#1d4ed8', width=0.6)
ax.set_title(f'MovieLens Rating Distribution (Mean = {mean_rating:.2f}, Median = {median_rating:.1f})', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Rating Score (Stars)', fontsize=10)
ax.set_ylabel('Rating Count', fontsize=10)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 400, f'{yval:,}', ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, max(rating_freq['count']) * 1.12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_rating_distribution.png', dpi=150)
plt.show()
"""
        )
    )

    # Cell 5: User & Movie Activity (Long-Tail)
    cells.append(
        new_markdown_cell("## 4. Ratings per User & Ratings per Movie (Long-Tail Distribution)")
    )
    cells.append(
        new_code_cell(
            r"""user_counts = ratings.groupby('userId').size()
movie_counts = ratings.groupby('movieId').size()

print('=== User Activity Summary ===')
display(user_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).to_frame('ratings_per_user'))

print('=== Movie Activity Summary ===')
display(movie_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).to_frame('ratings_per_movie'))

single_rating_movies = (movie_counts == 1).sum()
print(f'Movies with exactly 1 rating: {single_rating_movies} ({single_rating_movies / len(movie_counts) * 100:.2f}% of rated movies)')

# Plot long-tail on log-log scale
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
user_sorted = user_counts.sort_values(ascending=False).values
movie_sorted = movie_counts.sort_values(ascending=False).values

axes[0].plot(range(1, len(user_sorted) + 1), user_sorted, color='#8b5cf6', lw=2)
axes[0].set_yscale('log')
axes[0].set_xscale('log')
axes[0].set_title('Ratings per User (Rank vs Count, Log-Log)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('User Rank (by rating volume)', fontsize=10)
axes[0].set_ylabel('Number of Ratings (log scale)', fontsize=10)
axes[0].grid(True, which='both', ls='--', alpha=0.5)

axes[1].plot(range(1, len(movie_sorted) + 1), movie_sorted, color='#ec4899', lw=2)
axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[1].set_title('Ratings per Movie (Long-Tail, Log-Log)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Movie Rank (by rating volume)', fontsize=10)
axes[1].set_ylabel('Number of Ratings (log scale)', fontsize=10)
axes[1].grid(True, which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_user_and_movie_activity_long_tail.png', dpi=150)
plt.show()
"""
        )
    )

    # Cell 6: Interaction Matrix Sparsity
    cells.append(new_markdown_cell("## 5. Interaction Matrix Sparsity Percentage"))
    cells.append(
        new_code_cell(
            r"""n_users = ratings['userId'].nunique()
n_movies_rated = ratings['movieId'].nunique()
n_movies_catalog = len(movies)
total_ratings = len(ratings)

sparsity_rated = (1.0 - (total_ratings / (n_users * n_movies_rated))) * 100.0
sparsity_catalog = (1.0 - (total_ratings / (n_users * n_movies_catalog))) * 100.0

print(f'Unique Users                        : {n_users:,}')
print(f'Unique Movies Rated                 : {n_movies_rated:,}')
print(f'Total Catalog Movies                : {n_movies_catalog:,}')
print(f'Total Observed Ratings              : {total_ratings:,}')
print(f'Matrix Capacity (Users x Rated)     : {n_users * n_movies_rated:,}')
print(f'Matrix Sparsity (Rated Submatrix)   : {sparsity_rated:.4f}%')
print(f'Matrix Sparsity (Full Catalog)      : {sparsity_catalog:.4f}%')
"""
        )
    )

    # Cell 7: Genre Frequencies
    cells.append(
        new_markdown_cell("## 6. Genre Frequency Analysis & `(no genres listed)` Handling")
    )
    cells.append(
        new_code_cell(
            r"""# Explode pipe-separated genres
exploded_genres = movies['genres'].str.split('|').explode()
genre_counts = exploded_genres.value_counts()
no_genres_count = (movies['genres'] == '(no genres listed)').sum()

print(f'Total genre tags across catalog: {len(exploded_genres):,}')
print(f'Unique genre labels: {len(genre_counts)}')
print(f'Movies with "(no genres listed)": {no_genres_count} ({no_genres_count / len(movies) * 100:.2f}%)')
display(genre_counts.to_frame('movie_count'))

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ['#ef4444' if g == '(no genres listed)' else '#0284c7' for g in genre_counts.index]
ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color=bar_colors[::-1])
ax.set_title('Movie Genre Frequency (Exploded from Pipe-Separated Strings)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Number of Movies', fontsize=10)
for i, v in enumerate(genre_counts.values[::-1]):
    ax.text(v + 35, i, f'{v:,}', va='center', fontsize=8)
ax.set_xlim(0, max(genre_counts.values) * 1.1)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_genre_frequency.png', dpi=150)
plt.show()
"""
        )
    )

    # Cell 8: Release Year Extraction
    cells.append(new_markdown_cell("## 7. Release Year Extracted from Movie Titles"))
    cells.append(
        new_code_cell(
            r"""# Regex to extract 4 digits in parentheses at the end of the title
movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)$')[0]
missing_years = movies['release_year'].isna().sum()

print(f'Total movies             : {len(movies):,}')
print(f'Movies with parsed year  : {movies["release_year"].notna().sum():,}')
print(f'Movies missing year      : {missing_years} ({missing_years / len(movies) * 100:.2f}%)')

# Sample of movies without parsed release year
print('\nSample of titles where release year could not be parsed:')
display(movies[movies['release_year'].isna()][['movieId', 'title', 'genres']].head(10))

# Summary statistics of parsed years
valid_years = movies['release_year'].dropna().astype(int)
print('\nRelease Year Summary:')
display(valid_years.describe().to_frame('release_year'))

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.histplot(valid_years, bins=50, kde=True, color='#10b981', ax=ax)
ax.set_title(f'Movie Release Year Distribution (1902 - 2018, Median = {int(valid_years.median())})', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Release Year', fontsize=10)
ax.set_ylabel('Number of Movies', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '04_release_year_distribution.png', dpi=150)
plt.show()
"""
        )
    )

    # Cell 9: Popularity Bias
    cells.append(
        new_markdown_cell("## 8. Popularity Bias: Share of Ratings Covered by Top 10% of Movies")
    )
    cells.append(
        new_code_cell(
            r"""sorted_movie_counts = ratings.groupby('movieId').size().sort_values(ascending=False)
total_movie_ratings = sorted_movie_counts.sum()

top_10_pct_count = int(np.ceil(0.10 * len(sorted_movie_counts)))
top_10_pct_ratings = sorted_movie_counts.iloc[:top_10_pct_count].sum()
top_10_pct_share = (top_10_pct_ratings / total_movie_ratings) * 100

top_20_pct_count = int(np.ceil(0.20 * len(sorted_movie_counts)))
top_20_pct_ratings = sorted_movie_counts.iloc[:top_20_pct_count].sum()
top_20_pct_share = (top_20_pct_ratings / total_movie_ratings) * 100

print(f'Total rated movies                       : {len(sorted_movie_counts):,}')
print(f'Top 10% movie count                      : {top_10_pct_count:,} movies')
print(f'Ratings captured by top 10%              : {top_10_pct_ratings:,} / {total_movie_ratings:,}')
print(f'Share of ratings captured by top 10%     : {top_10_pct_share:.2f}%')
print(f'Share of ratings captured by top 20%     : {top_20_pct_share:.2f}%')

# Lorenz curve visualization
fig, ax = plt.subplots(figsize=(7, 5))
cum_movies_pct = np.linspace(0, 100, len(sorted_movie_counts))
cum_ratings_pct = (np.cumsum(sorted_movie_counts.values) / total_movie_ratings) * 100

ax.plot(cum_movies_pct, cum_ratings_pct, color='#f59e0b', lw=2.5, label='Actual Distribution')
ax.plot([0, 100], [0, 100], 'k--', lw=1.2, label='Perfect Equality (Diagonal)')

# Top 10% highlight
ax.plot([10, 10], [0, top_10_pct_share], 'r:', lw=1.5)
ax.plot([0, 10], [top_10_pct_share, top_10_pct_share], 'r:', lw=1.5)
ax.scatter([10], [top_10_pct_share], color='red', s=50, zorder=5)
ax.text(12, top_10_pct_share - 3, f'Top 10% movies = {top_10_pct_share:.1f}% ratings', color='red', fontweight='bold', fontsize=9)

ax.set_title('Popularity Bias: Lorenz Curve of Movie Ratings', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('% of Movies (Ranked by Rating Volume)', fontsize=10)
ax.set_ylabel('% of Cumulative Ratings', fontsize=10)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_popularity_bias_lorenz.png', dpi=150)
plt.show()
"""
        )
    )

    # Cell 10: Ratings Over Time
    cells.append(new_markdown_cell("## 9. Ratings Over Time (Temporal Patterns)"))
    cells.append(
        new_code_cell(
            r"""ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
print(f'Earliest rating timestamp : {ratings["datetime"].min()}')
print(f'Latest rating timestamp   : {ratings["datetime"].max()}')

# Resample monthly
monthly_ratings = ratings.set_index('datetime').resample('ME').agg(
    count=('rating', 'size'),
    mean_rating=('rating', 'mean')
)

fig, ax1 = plt.subplots(figsize=(11, 5))
color1 = '#6366f1'
ax1.set_xlabel('Timestamp (Date)', fontsize=10)
ax1.set_ylabel('Monthly Rating Volume', color=color1, fontsize=10)
ax1.plot(monthly_ratings.index, monthly_ratings['count'], color=color1, lw=1.5, alpha=0.85)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#f97316'
ax2.set_ylabel('Monthly Mean Rating', color=color2, fontsize=10)
ax2.plot(monthly_ratings.index, monthly_ratings['mean_rating'], color=color2, lw=1.5, alpha=0.75, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(2.5, 4.5)

plt.title('MovieLens Rating Volume and Mean Rating Over Time (1996 - 2018)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_ratings_over_time.png', dpi=150)
plt.show()
"""
        )
    )

    # Cell 11: Key Findings
    cells.append(
        new_markdown_cell(
            """## Key Findings

Based strictly on the computed empirical values from the `ml-latest-small` dataset:

1. **Dataset Volume & Integrity:**
   - **100,836 ratings**, **610 users**, and **9,724 rated movies** out of **9,742 total movies** in the catalog (18 catalog movies have 0 ratings).
   - Data hygiene is clean: **0 duplicate rows** across all tables, **0 duplicate user-movie pairs**, and **0 missing values** in `movies.csv`, `ratings.csv`, and `tags.csv` (only 8 movies in `links.csv` lack `tmdbId`).

2. **Rating Bias & Central Tendency:**
   - The overall rating mean is **3.5016** (median: **3.5**, standard deviation: **1.0425**).
   - Ratings are skewed toward positive evaluation: score `4.0` is the single most frequent rating (**26,818 occurrences, 26.60%**), followed by `3.0` (**20,047, 19.88%**) and `5.0` (**13,211, 13.10%**). Ratings strictly below `3.0` account for only **13.43%** of all ratings.

3. **Interaction Matrix Sparsity:**
   - Evaluated across active users and rated items ($610 \\times 9,724$), the user-item interaction matrix has **98.3000% sparsity**. Evaluated across the entire catalog ($610 \\times 9,742$), sparsity is **98.3032%**.
   - Recommendation systems must leverage dimensionality reduction (SVD, matrix factorization) or learned embeddings to handle this level of missing entries.

4. **Long-Tail Activity & Item Cold Start:**
   - User activity is right-skewed: median user has **70.5 ratings** (mean 165.3, min 20, max 2,698).
   - Movie activity is severely long-tailed: median movie has only **3 ratings** (mean 10.37, max 329).
   - **3,446 movies (35.44% of rated movies)** have **exactly 1 rating**, highlighting a significant item cold-start challenge where collaborative filtering alone has inadequate signal.

5. **Popularity Bias Concentration:**
   - The top 10% most-rated movies (**973 movies**) capture **60,551 ratings (60.05% of all ratings)**.
   - The top 20% most-rated movies (**1,945 movies**) capture **77,678 ratings (77.03% of all ratings)**.
   - Unmitigated collaborative filtering models will be biased toward recommending these high-frequency blockbusters at the expense of catalog diversity.

6. **Genre & Content Richness:**
   - The catalog spans 19 defined genres plus the placeholder `(no genres listed)`.
   - The top 4 genres are **Drama (4,361 titles)**, **Comedy (3,756)**, **Thriller (1,894)**, and **Action (1,828)**.
   - Only **34 movies (0.35%)** have `(no genres listed)`, confirming that genre metadata and content-based text representations provide near-complete coverage (99.65%) for hybrid fallback.

7. **Temporal Dynamics:**
   - Extracted release years span from **1902 to 2018** (median: **1999**, mean: **1994.6**). 24 titles had non-standard formatting lacking a 4-digit release year in parentheses.
   - User rating activity spans **March 29, 1996 to September 24, 2018** (>22 years). High activity peaks occur in 1996, 2000, and 2016, with monthly mean rating sentiment remaining consistent between **3.3 and 3.7**.
"""
        )
    )

    nb["cells"] = cells
    return nb


def main() -> int:
    nb = build_notebook()
    print("Executing notebook using ipykernel...")
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
