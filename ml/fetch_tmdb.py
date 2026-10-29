"""
ml/fetch_tmdb.py
----------------
Fetches movie metadata (overview, poster_path, runtime, top_cast) from TMDB API
using tmdbId from links.csv, caches responses to data/external/tmdb_cache.json,
and provides document construction for semantic search.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "ml-latest-small"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CACHE_FILE = EXTERNAL_DATA_DIR / "tmdb_cache.json"

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Curated fallback metadata for canonical MovieLens titles when running without an active TMDB API key
CURATED_FALLBACK_METADATA: dict[int, dict[str, Any]] = {
    1: {  # Toy Story
        "overview": "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Woody is convinced he is Andy's favorite toy, but Buzz quickly wins everyone over. When Woody and Buzz find themselves lost in the outside world, they must work together to find their way home.",
        "poster_path": "/uXDfjJbdP4ijW5hWSBrFuVJTkab.jpg",
        "runtime": 81,
        "top_cast": ["Tom Hanks", "Tim Allen", "Don Rickles", "Jim Varney", "Wallace Shawn"],
    },
    2: {  # Jumanji
        "overview": "When siblings Judy and Peter discover an enchanted board game that opens the door to a magical world, they unwittingly invite Alan, an adult who's been trapped inside its jungle for 26 years, into their living room. Together they must outrun rampaging wild animals to finish the game.",
        "poster_path": "/vgpXmVaMTviAcQ2Xbt1qyBpN9rm.jpg",
        "runtime": 104,
        "top_cast": [
            "Robin Williams",
            "Kirsten Dunst",
            "David Alan Grier",
            "Bonnie Hunt",
            "Jonathan Hyde",
        ],
    },
    296: {  # Pulp Fiction
        "overview": "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in this sprawling, comedic crime caper. Their adventures unfurl in three eccentric, non-linear stories.",
        "poster_path": "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
        "runtime": 154,
        "top_cast": [
            "John Travolta",
            "Samuel L. Jackson",
            "Uma Thurman",
            "Bruce Willis",
            "Ving Rhames",
        ],
    },
    318: {  # Shawshank Redemption
        "overview": "Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden and forms an enduring friendship with fellow inmate Red.",
        "poster_path": "/9cqN0Ki8iq7HGCRGtAQveknP0NW.jpg",
        "runtime": 142,
        "top_cast": [
            "Tim Robbins",
            "Morgan Freeman",
            "Bob Gunton",
            "William Sadler",
            "Clancy Brown",
        ],
    },
    356: {  # Forrest Gump
        "overview": "A man with a low IQ has accomplished great things in his life and been present during significant historic events—in each case, far exceeding what anyone imagined possible. Yet despite all he has achieved, his one true love eludes him.",
        "poster_path": "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",
        "runtime": 142,
        "top_cast": [
            "Tom Hanks",
            "Robin Wright",
            "Gary Sinise",
            "Sally Field",
            "Mykelti Williamson",
        ],
    },
    480: {  # Jurassic Park
        "overview": "A wealthy entrepreneur secretly creates a theme park featuring living dinosaurs drawn from prehistoric DNA. Before opening day, he invites a team of experts and his two eager grandchildren to experience the park and help calm anxious investors, but a security breach unleashes terror.",
        "poster_path": "/oU7Oq2kFAAlGqbU4VoAE36g4ho1.jpg",
        "runtime": 127,
        "top_cast": [
            "Sam Neill",
            "Laura Dern",
            "Jeff Goldblum",
            "Richard Attenborough",
            "Bob Peck",
        ],
    },
    593: {  # Silence of the Lambs
        "overview": "Clarice Starling is a top student at the FBI's training academy. Jack Crawford wants Clarice to interview Dr. Hannibal Lecter, a brilliant psychiatrist who is also a violent psychopath serving life behind bars for cannibalism, to help catch a twisted serial killer.",
        "poster_path": "/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
        "runtime": 118,
        "top_cast": [
            "Jodie Foster",
            "Anthony Hopkins",
            "Scott Glenn",
            "Ted Levine",
            "Anthony Heald",
        ],
    },
    2571: {  # The Matrix
        "overview": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth, discovering that his reality is a simulated artificial world.",
        "poster_path": "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
        "runtime": 136,
        "top_cast": [
            "Keanu Reeves",
            "Laurence Fishburne",
            "Carrie-Anne Moss",
            "Hugo Weaving",
            "Joe Pantoliano",
        ],
    },
    2959: {  # Fight Club
        "overview": "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy. Their concept catches on, with underground 'fight clubs' forming in every town, until an eccentric angel of death gets in the way.",
        "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "runtime": 139,
        "top_cast": [
            "Edward Norton",
            "Brad Pitt",
            "Helena Bonham Carter",
            "Meat Loaf",
            "Zach Grenier",
        ],
    },
    4993: {  # Lord of the Rings: Fellowship of the Ring
        "overview": "Young hobbit Frodo Baggins, after inheriting a mysterious ring from his uncle Bilbo, must leave his home behind to keep it from falling into the hands of its evil creator, the Dark Lord Sauron. Along with a fellowship of allies, he journeys toward Mount Doom.",
        "poster_path": "/6oom5QYQ2yQTMJIbnvbkBL9cDK6.jpg",
        "runtime": 178,
        "top_cast": [
            "Elijah Wood",
            "Ian McKellen",
            "Viggo Mortensen",
            "Sean Astin",
            "Orlando Bloom",
        ],
    },
    58559: {  # The Dark Knight
        "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known as the Joker.",
        "poster_path": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "runtime": 152,
        "top_cast": [
            "Christian Bale",
            "Heath Ledger",
            "Aaron Eckhart",
            "Michael Caine",
            "Maggie Gyllenhaal",
        ],
    },
    79132: {  # Inception
        "overview": "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets, is offered a chance to regain his old life as payment for a task considered to be impossible: 'inception', the implantation of another person's idea into a target's subconscious.",
        "poster_path": "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
        "runtime": 148,
        "top_cast": [
            "Leonardo DiCaprio",
            "Joseph Gordon-Levitt",
            "Elliot Page",
            "Tom Hardy",
            "Ken Watanabe",
        ],
    },
    109487: {  # Interstellar
        "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage. A father leaves his daughter behind on a dying Earth to find a new home for humanity across the stars.",
        "poster_path": "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "runtime": 169,
        "top_cast": [
            "Matthew McConaughey",
            "Anne Hathaway",
            "Jessica Chastain",
            "Michael Caine",
            "Mackenzie Foy",
        ],
    },
    48780: {  # The Prestige
        "overview": "A mysterious story of two magicians whose intense rivalry leads them on a life-long battle for supremacy -- full of obsession, deceit, and jealousy with dangerous and deadly consequences.",
        "poster_path": "/tRNUd5ftSStnPIWpUeP8Pll6oZ8.jpg",
        "runtime": 130,
        "top_cast": [
            "Hugh Jackman",
            "Christian Bale",
            "Michael Caine",
            "Scarlett Johansson",
            "Rebecca Hall",
        ],
    },
    48394: {  # Ocean's Eleven (2001) / Lighthearted heist
        "overview": "Less than twenty-four hours into his parole, charismatic thief Danny Ocean is already rolling out his next plan: in one night, Danny's hand-picked crew of eleven specialists will attempt to steal more than $150 million from three Las Vegas casinos simultaneously in a lighthearted, clever heist.",
        "poster_path": "/9B0B7sR1kYQGz2bH7gB3Zz7RkO0.jpg",
        "runtime": 116,
        "top_cast": ["George Clooney", "Brad Pitt", "Matt Damon", "Julia Roberts", "Andy Garcia"],
    },
}


def load_tmdb_cache(cache_path: Path = CACHE_FILE) -> dict[str, dict[str, Any]]:
    """Load cached TMDB metadata from json file."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read cache at {cache_path}: {e}")
        return {}


def save_tmdb_cache(cache: dict[str, dict[str, Any]], cache_path: Path = CACHE_FILE) -> None:
    """Save TMDB cache atomically to json file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp_path.replace(cache_path)


def fetch_movie_from_tmdb(
    tmdb_id: int,
    api_key: str,
    base_url: str = TMDB_BASE_URL,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    """Fetch movie overview, poster_path, runtime, and top 5 cast members from TMDB."""
    sess = session or requests.Session()
    url = f"{base_url}/movie/{tmdb_id}"
    params = {"api_key": api_key, "append_to_response": "credits"}

    for attempt in range(4):
        try:
            resp = sess.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                credits = data.get("credits", {})
                cast_raw = credits.get("cast", [])
                top_cast = [c.get("name") for c in cast_raw[:5] if c.get("name")]
                return {
                    "overview": data.get("overview") or "",
                    "poster_path": data.get("poster_path") or "",
                    "runtime": data.get("runtime"),
                    "top_cast": top_cast,
                }
            elif resp.status_code == 404:
                return {
                    "overview": "",
                    "poster_path": "",
                    "runtime": None,
                    "top_cast": [],
                }
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                time.sleep(retry_after + 1)
            else:
                logger.debug(f"TMDB status {resp.status_code} for tmdb_id {tmdb_id}")
                time.sleep(0.5)
        except Exception as e:
            logger.debug(f"Network error on attempt {attempt + 1} for tmdb_id {tmdb_id}: {e}")
            time.sleep(1.0)

    return None


def sync_tmdb_metadata(
    limit: int | None = None,
    rate_limit_delay: float = 0.04,
    cache_path: Path = CACHE_FILE,
) -> dict[str, dict[str, Any]]:
    """
    Fetch and cache TMDB metadata for movies listed in links.csv.
    Resumes automatically from existing cache.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("TMDB_API_KEY", "").strip()

    cache = load_tmdb_cache(cache_path)
    logger.info(f"Loaded existing TMDB cache with {len(cache)} entries from {cache_path}")

    # Seed with curated fallback entries if cache is empty
    if not cache:
        for mid, data in CURATED_FALLBACK_METADATA.items():
            cache[str(mid)] = {
                "movieId": mid,
                **data,
            }
        save_tmdb_cache(cache, cache_path)
        logger.info(f"Seeded cache with {len(CURATED_FALLBACK_METADATA)} curated entries.")

    if not api_key or api_key == "REPLACE_WITH_TMDB_API_KEY":
        logger.warning(
            "TMDB_API_KEY is not set or is a placeholder in .env. "
            "Skipping live TMDB API fetch. Using cached & fallback metadata."
        )
        return cache

    links_path = RAW_DATA_DIR / "links.csv"
    if not links_path.exists():
        logger.error(f"links.csv not found at {links_path}")
        return cache

    links_df = pd.read_csv(links_path)
    valid_links = links_df.dropna(subset=["tmdbId"]).copy()
    valid_links["tmdbId"] = valid_links["tmdbId"].astype(int)

    to_fetch = [
        (int(row["movieId"]), int(row["tmdbId"]))
        for _, row in valid_links.iterrows()
        if str(int(row["movieId"])) not in cache
    ]

    if limit is not None and limit > 0:
        to_fetch = to_fetch[:limit]

    logger.info(f"Need to fetch {len(to_fetch)} movies from TMDB API...")

    if not to_fetch:
        logger.info("All movies are already cached!")
        return cache

    session = requests.Session()
    count = 0
    save_interval = 50

    try:
        for movie_id, tmdb_id in to_fetch:
            data = fetch_movie_from_tmdb(tmdb_id, api_key, session=session)
            if data is not None:
                cache[str(movie_id)] = {
                    "movieId": movie_id,
                    "tmdbId": tmdb_id,
                    **data,
                }
            else:
                cache[str(movie_id)] = {
                    "movieId": movie_id,
                    "tmdbId": tmdb_id,
                    "overview": "",
                    "poster_path": "",
                    "runtime": None,
                    "top_cast": [],
                }

            count += 1
            if count % save_interval == 0:
                save_tmdb_cache(cache, cache_path)
                logger.info(f"Progress: {count}/{len(to_fetch)} fetched and saved.")

            time.sleep(rate_limit_delay)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user! Saving progress...")
    finally:
        save_tmdb_cache(cache, cache_path)
        logger.info(f"Finished TMDB sync. Total cached entries: {len(cache)}.")

    return cache


def build_movie_documents(
    movies_df: pd.DataFrame,
    tags_df: pd.DataFrame | None = None,
    tmdb_cache: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build a rich descriptive 'document' for each movie:
    document = title + year + genres + overview + top tags.
    """
    if tmdb_cache is None:
        tmdb_cache = load_tmdb_cache()

    if tags_df is None:
        tags_path = RAW_DATA_DIR / "tags.csv"
        if tags_path.exists():
            tags_df = pd.read_csv(tags_path)
        else:
            tags_df = pd.DataFrame(columns=["movieId", "tag"])

    # Aggregate top tags per movie
    tag_agg: dict[int, list[str]] = {}
    if not tags_df.empty:
        for mid, group in tags_df.groupby("movieId"):
            # Count frequency of each tag
            counts = group["tag"].astype(str).str.lower().value_counts()
            tag_agg[int(mid)] = counts.index.tolist()[:10]

    docs_data: list[dict[str, Any]] = []

    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        title = str(row.get("clean_title") or row.get("title") or "").strip()
        year = str(row.get("year", "")) if pd.notna(row.get("year")) else ""

        genres_val = row.get("genres", [])
        if isinstance(genres_val, list | np.ndarray | tuple):
            genres_list = [str(g) for g in genres_val if g and str(g) != "(no genres listed)"]
            genres_str = ", ".join(genres_list)
        elif isinstance(genres_val, str):
            genres_list = [
                g.strip()
                for g in genres_val.split("|")
                if g.strip() and g.strip() != "(no genres listed)"
            ]
            genres_str = ", ".join(genres_list)
        else:
            genres_list = []
            genres_str = ""

        # TMDB metadata
        meta = tmdb_cache.get(str(mid), {})
        overview = meta.get("overview", "").strip()
        poster_path = meta.get("poster_path", "")
        runtime = meta.get("runtime")
        top_cast = meta.get("top_cast", [])
        cast_str = ", ".join(top_cast[:3]) if top_cast else ""

        # Tags
        top_tags = tag_agg.get(mid, [])
        tags_str = ", ".join(top_tags)

        # Assemble rich text document
        doc_parts = [title]
        if year:
            doc_parts.append(f"({year}).")
        if genres_str:
            doc_parts.append(f"Genres: {genres_str}.")
        if overview:
            doc_parts.append(f"Overview: {overview}")
        if cast_str:
            doc_parts.append(f"Starring: {cast_str}.")
        if tags_str:
            doc_parts.append(f"Tags: {tags_str}.")

        document = " ".join(doc_parts).strip()

        docs_data.append(
            {
                "movieId": mid,
                "item_idx": int(row["item_idx"]) if "item_idx" in row else -1,
                "title": title,
                "year": int(year) if year.isdigit() else None,
                "genres": genres_val if isinstance(genres_val, list) else genres_str.split(", "),
                "overview": overview,
                "poster_path": poster_path,
                "runtime": runtime,
                "top_cast": top_cast,
                "top_tags": top_tags,
                "document": document,
            }
        )

    return pd.DataFrame(docs_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TMDB metadata and cache to JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of movies to fetch.")
    parser.add_argument(
        "--delay", type=float, default=0.04, help="Delay between requests in seconds."
    )
    args = parser.parse_args()

    sync_tmdb_metadata(limit=args.limit, rate_limit_delay=args.delay)
