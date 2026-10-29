"""
scripts/enrich_posters.py
-------------------------
Fetches real movie posters for popular and homepage movies using IMDb IDs from
links.csv and OMDb API, which works reliably globally (including in India) without
requiring a TMDB account or VPN.
"""

import json
import logging
import sqlite3
import time
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "cinematch.db"
LINKS_CSV = PROJECT_ROOT / "data" / "raw" / "ml-latest-small" / "links.csv"
RATINGS_CSV = PROJECT_ROOT / "data" / "raw" / "ml-latest-small" / "ratings.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "external" / "imdb_poster_cache.json"

OMDB_KEYS = ["trilogy", "b7da8d63", "2b0907e5"]


def fetch_poster_from_omdb(imdb_id: str) -> str | None:
    for key in OMDB_KEYS:
        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                poster = data.get("Poster")
                if poster and poster != "N/A" and poster.startswith("http"):
                    return poster
        except Exception:
            continue
    return None


def main(limit: int = 120):
    logger.info("Reading links and ratings datasets...")
    links_df = pd.read_csv(LINKS_CSV).set_index("movieId")
    ratings_df = pd.read_csv(RATINGS_CSV)

    # Find most popular movies by rating count
    top_mids = ratings_df["movieId"].value_counts().head(limit).index.tolist()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Also include movies currently displayed on homepage genre rows (first 10 of Sci-Fi, Action, Animation, Drama)
    genre_movies = cur.execute("""
        SELECT ml_movie_id FROM movies 
        WHERE (genres LIKE '%Action%' OR genres LIKE '%Sci-Fi%' OR genres LIKE '%Animation%' OR genres LIKE '%Drama%')
        LIMIT 60
    """).fetchall()

    target_mids = list(dict.fromkeys(top_mids + [row[0] for row in genre_movies]))
    logger.info(f"Identified {len(target_mids)} target movies to enrich.")

    # Load cache if exists
    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cache = {}

    success_count = 0
    for idx, mid in enumerate(target_mids, start=1):
        if mid not in links_df.index:
            continue

        raw_imdb = links_df.loc[mid, "imdbId"]
        if pd.isna(raw_imdb):
            continue

        imdb_id = f"tt{str(int(raw_imdb)).zfill(7)}"

        # Check existing db value
        existing = cur.execute(
            "SELECT poster_path FROM movies WHERE ml_movie_id = ?", (mid,)
        ).fetchone()
        if existing and existing[0] and existing[0].startswith("http"):
            continue

        poster_url = cache.get(imdb_id)
        if not poster_url:
            poster_url = fetch_poster_from_omdb(imdb_id)
            if poster_url:
                cache[imdb_id] = poster_url
            time.sleep(0.1)

        if poster_url:
            cur.execute(
                "UPDATE movies SET poster_path = ? WHERE ml_movie_id = ?", (poster_url, mid)
            )
            success_count += 1
            if success_count % 10 == 0 or idx == len(target_mids):
                con.commit()
                logger.info(
                    f"Progress [{idx}/{len(target_mids)}]: Enriched {success_count} movie posters so far."
                )

    con.commit()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    logger.info(f"Finished! Successfully enriched {success_count} movies with posters.")


if __name__ == "__main__":
    main()
