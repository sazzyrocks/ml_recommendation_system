import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.hybrid_model import HybridRecommender

train_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "train.parquet")
h = HybridRecommender()
h.fit(train_df)

print("=" * 70)
print("DEMO 1: NEW USER WITH 0 RATINGS + ONBOARDING GENRE PREFERENCES ['Sci-Fi', 'Action']")
print("=" * 70)
recs_sci_fi = h.recommend_cold_user(genre_preferences=["Sci-Fi", "Action"], k=5)
for r in recs_sci_fi:
    print(
        f"[{r['top_source'].upper()}] {r['title']} ({', '.join(r['genres'][:3])}) -> Reason: {r['reason']}"
    )

print("\n" + "=" * 70)
print("DEMO 2: USER WITH 1-5 SPARSE RATINGS (Rated: Toy Story 5.0, Jumanji 4.0)")
print("=" * 70)
# item_idx 0 is Toy Story, item_idx 1 is Jumanji
recs_sparse = h.recommend_sparse_user({0: 5.0, 1: 4.0}, k=5)
for r in recs_sparse:
    print(
        f"[{r['top_source'].upper()}] {r['title']} ({', '.join(r['genres'][:3])}) -> Reason: {r['reason']}"
    )

print("\n" + "=" * 70)
print("DEMO 3: NEW MOVIE WITH 0 RATINGS (Assessing compatibility via Content / Genres)")
print("=" * 70)
# Movie with genres ['Sci-Fi', 'Action'] tested for user 0
score_new = h.score_new_movie(user_idx=0, genres=["Sci-Fi", "Action", "Adventure"])
print(f"Compatibility score for new unrated Sci-Fi/Action film for User 0: {score_new:.4f}")

print("\n" + "=" * 70)
print("DEMO 4: FOLD-IN FOR ADAPTIVE PERSONALIZATION (Rated: Inception 5.0, Interstellar 5.0)")
print("=" * 70)
# Find item_idx for Inception and Interstellar
mid_to_item = h.mid_to_item_idx
inception_idx = mid_to_item.get(79132)  # Inception
interstellar_idx = mid_to_item.get(109487)  # Interstellar
print(f"Inception item_idx: {inception_idx}, Interstellar item_idx: {interstellar_idx}")

recs_fold = h.recommend_fold_in_user({inception_idx: 5.0, interstellar_idx: 5.0}, k=5)
for r in recs_fold:
    print(
        f"[{r['top_source'].upper()}] {r['title']} ({', '.join(r['genres'][:3])}) -> Reason: {r['reason']}"
    )
print("=" * 70)
