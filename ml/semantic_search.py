"""
ml/semantic_search.py
---------------------
Natural language semantic movie search using sentence-transformers (all-MiniLM-L6-v2).
Builds 384-dimensional normalized dense embeddings over rich movie documents,
supports fast cosine similarity search with optional genre and release year filters,
and serializes embeddings to artifacts/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.fetch_tmdb import build_movie_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "movie_embeddings.npy"
METADATA_PATH = ARTIFACTS_DIR / "semantic_movies.parquet"


class SemanticSearch:
    """Semantic search engine for movies based on dense sentence embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        embeddings_path: Path = EMBEDDINGS_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self.model_name = model_name
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)

        self._model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.metadata_df: pd.DataFrame | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(
        self,
        movies_df: pd.DataFrame | None = None,
        force: bool = False,
        batch_size: int = 64,
    ) -> None:
        """
        Build and save dense embeddings for all movies.
        """
        if not force and self.embeddings_path.exists() and self.metadata_path.exists():
            logger.info("Found existing embeddings and metadata artifacts. Loading from disk...")
            self.load_index()
            return

        logger.info("Building semantic search documents and embeddings...")
        if movies_df is None:
            cleaned_path = PROCESSED_DATA_DIR / "movies_cleaned.parquet"
            if not cleaned_path.exists():
                raise FileNotFoundError(f"movies_cleaned.parquet not found at {cleaned_path}")
            movies_df = pd.read_parquet(cleaned_path)

        # Build rich text documents
        docs_df = build_movie_documents(movies_df)
        documents = docs_df["document"].tolist()

        logger.info(f"Encoding {len(documents)} movie documents with {self.model_name}...")
        # SentenceTransformer encode with normalization returns unit L2 vectors
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        # Ensure artifacts directory exists
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Saving embeddings (shape: {self.embeddings.shape}) to {self.embeddings_path}..."
        )
        np.save(self.embeddings_path, self.embeddings)

        logger.info(f"Saving metadata to {self.metadata_path}...")
        docs_df.to_parquet(self.metadata_path, index=False)
        self.metadata_df = docs_df

    def load_index(self) -> None:
        """Load pre-computed embeddings and metadata from artifacts/."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        logger.info(f"Loading embeddings from {self.embeddings_path}...")
        self.embeddings = np.load(self.embeddings_path)
        logger.info(f"Loading metadata from {self.metadata_path}...")
        self.metadata_df = pd.read_parquet(self.metadata_path)

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search movies using natural language semantic similarity.

        Args:
            query: Natural language search string.
            k: Number of top results to return.
            filters: Optional dictionary containing filter criteria:
                     - "genre": str (e.g. "Sci-Fi", "Comedy")
                     - "year_min": int (inclusive lower year bound)
                     - "year_max": int (inclusive upper year bound)
                     - "year_range": tuple[int, int] (e.g. (1990, 2005))

        Returns:
            List of result dictionaries sorted in descending order of similarity.
        """
        if self.embeddings is None or self.metadata_df is None:
            self.load_index()

        # 1. Encode user query
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]

        # 2. Compute cosine similarity (dot product of L2-normalized vectors)
        similarities = np.dot(self.embeddings, query_vector)

        # 3. Apply optional filters
        valid_mask = np.ones(len(self.metadata_df), dtype=bool)

        if filters:
            # Genre filter
            genre_filter = filters.get("genre")
            if genre_filter:
                genre_lower = str(genre_filter).strip().lower()
                genre_matches = self.metadata_df["genres"].apply(
                    lambda g_list: any(str(g).lower() == genre_lower for g in g_list)
                    if isinstance(g_list, list | np.ndarray)
                    else genre_lower in str(g_list).lower()
                )
                valid_mask &= genre_matches.to_numpy()

            # Year range filter
            year_range = filters.get("year_range")
            if year_range and len(year_range) == 2:
                y_min, y_max = year_range
                valid_mask &= (self.metadata_df["year"] >= y_min) & (
                    self.metadata_df["year"] <= y_max
                )
            else:
                y_min = filters.get("year_min")
                if y_min is not None:
                    valid_mask &= self.metadata_df["year"] >= y_min
                y_max = filters.get("year_max")
                if y_max is not None:
                    valid_mask &= self.metadata_df["year"] <= y_max

        # 4. Rank candidates
        candidate_indices = np.where(valid_mask)[0]
        if len(candidate_indices) == 0:
            return []

        cand_sims = similarities[candidate_indices]
        if len(candidate_indices) > k:
            # Partial sort for speed
            top_cand_sub = np.argpartition(cand_sims, -k)[-k:]
            top_cand_sub = top_cand_sub[np.argsort(-cand_sims[top_cand_sub])]
            top_indices = candidate_indices[top_cand_sub]
        else:
            top_indices = candidate_indices[np.argsort(-cand_sims)]

        # 5. Format results
        results: list[dict[str, Any]] = []
        for idx in top_indices:
            row = self.metadata_df.iloc[idx]
            sim_score = float(similarities[idx])
            genres_val = row["genres"]
            genres_list = (
                list(genres_val)
                if isinstance(genres_val, list | np.ndarray)
                else [str(genres_val)]
            )

            results.append(
                {
                    "movieId": int(row["movieId"]),
                    "item_idx": int(row.get("item_idx", -1)),
                    "title": str(row["title"]),
                    "year": int(row["year"]) if pd.notna(row["year"]) else None,
                    "genres": genres_list,
                    "similarity": round(sim_score, 4),
                    "overview": str(row.get("overview") or ""),
                    "top_cast": list(row.get("top_cast", []))
                    if isinstance(row.get("top_cast"), list | np.ndarray)
                    else [],
                    "document": str(row.get("document") or ""),
                }
            )

        return results


def run_benchmark_queries(output_path: Path = REPORTS_DIR / "semantic_search_examples.md") -> None:
    """
    Execute 10 diverse natural-language test queries and record results in Markdown format.
    """
    engine = SemanticSearch()
    engine.load_index()

    test_queries = [
        {
            "query": "space exploration with emotional father-daughter story",
            "filters": None,
            "category": "Theme & Emotion",
        },
        {
            "query": "lighthearted heist comedy",
            "filters": {"genre": "Comedy"},
            "category": "Genre & Plot Tropes",
        },
        {
            "query": "mind-bending psychological thriller with unreliable narrator",
            "filters": {"genre": "Thriller"},
            "category": "Psychological / Narrative Device",
        },
        {
            "query": "animated fantasy adventure for the whole family",
            "filters": {"genre": "Animation"},
            "category": "Tone & Audience",
        },
        {
            "query": "gritty crime noir with corrupt detectives in the city",
            "filters": {"genre": "Crime"},
            "category": "Atmosphere / Style",
        },
        {
            "query": "romantic comedy set in New York City",
            "filters": {"genre": "Romance"},
            "category": "Setting & Genre",
        },
        {
            "query": "time travel paradox and butterfly effect",
            "filters": {"genre": "Sci-Fi"},
            "category": "Sci-Fi Concept",
        },
        {
            "query": "post-apocalyptic survival in a desolate wasteland",
            "filters": None,
            "category": "World Building & Theme",
        },
        {
            "query": "inspiring biographical sports drama about underdog victory",
            "filters": {"genre": "Drama"},
            "category": "Narrative Arc & Mood",
        },
        {
            "query": "high-stakes courtroom drama with sharp dialogue",
            "filters": {"year_range": (1990, 2010)},
            "category": "Subgenre with Year Range Filter (1990-2010)",
        },
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Semantic Search Benchmark Examples",
        "",
        "> Natural language semantic search evaluation using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) on MovieLens + TMDB rich text documents.",
        "",
        "## Summary of Capabilities",
        "- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized 384-dimensional dense vectors)",
        "- **Retrieval Metric:** Cosine similarity via matrix-vector dot product",
        "- **Filtering:** Exact genre filtering and release year interval bounds",
        "- **Document Representation:** Title + Release Year + Curated Genres + TMDB Plot Overview + Top User Tags",
        "",
        "---",
        "",
    ]

    for i, tq in enumerate(test_queries, 1):
        q = tq["query"]
        filters = tq["filters"]
        cat = tq["category"]
        results = engine.search(q, k=5, filters=filters)

        filter_str = f" `{filters}`" if filters else " *None (unrestricted catalog)*"
        lines.append(f'### Query {i}: "{q}"')
        lines.append(f"- **Category:** {cat}")
        lines.append(f"- **Filters:** {filter_str}")
        lines.append("")
        lines.append("| Rank | Title | Year | Genres | Similarity | Synopsis / Overview Snippet |")
        lines.append("| :--- | :--- | :--- | :--- | :---: | :--- |")

        for rank, res in enumerate(results, 1):
            title = res["title"]
            yr = res["year"] if res["year"] else "N/A"
            genres = ", ".join(res["genres"][:3])
            sim = f"{res['similarity']:.4f}"
            ov = res["overview"]
            if not ov:
                ov = res["document"][:100] + "..."
            elif len(ov) > 110:
                ov = ov[:110] + "..."
            # Clean pipe characters for markdown table
            ov_clean = ov.replace("|", "/")
            lines.append(f"| **#{rank}** | **{title}** | {yr} | {genres} | `{sim}` | {ov_clean} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved semantic search benchmark report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Semantic Search Engine.")
    parser.add_argument("--build", action="store_true", help="Build or rebuild dense embeddings.")
    parser.add_argument("--query", type=str, default=None, help="Search query string.")
    parser.add_argument("--genre", type=str, default=None, help="Genre filter.")
    parser.add_argument("--year-min", type=int, default=None, help="Min release year.")
    parser.add_argument("--year-max", type=int, default=None, help="Max release year.")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run 10 test queries and save report."
    )
    args = parser.parse_args()

    engine = SemanticSearch()
    if args.build or not EMBEDDINGS_PATH.exists():
        engine.build_index(force=args.build)

    if args.benchmark:
        run_benchmark_queries()
    elif args.query:
        filters = {}
        if args.genre:
            filters["genre"] = args.genre
        if args.year_min is not None:
            filters["year_min"] = args.year_min
        if args.year_max is not None:
            filters["year_max"] = args.year_max

        res = engine.search(args.query, k=5, filters=filters if filters else None)
        print(f"\nTop results for '{args.query}':")
        for r in res:
            print(
                f"[{r['similarity']:.4f}] {r['title']} ({r['year']}) - Genres: {', '.join(r['genres'])}"
            )
            if r["overview"]:
                print(f"       Overview: {r['overview'][:120]}...")
