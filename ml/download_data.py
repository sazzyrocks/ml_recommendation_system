"""
ml/download_data.py
-------------------
Downloads and extracts the MovieLens ml-latest-small dataset into data/raw/.

Usage:
    python ml/download_data.py          # from project root
    python -m ml.download_data          # as a module

Idempotent: skips download if the zip is already present, skips extraction
if the target directory already exists.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_NAME = "ml-latest-small"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / f"{DATASET_NAME}.zip"
EXTRACTED_DIR = RAW_DIR / DATASET_NAME

# Expected files after extraction
EXPECTED_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv", "README.txt"]


def _download(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream-download *url* into *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DL] Downloading {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as fh,
        tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"[OK] Saved to {dest}")


def _extract(zip_path: Path, dest_dir: Path) -> None:
    """Extract *zip_path* into *dest_dir*."""
    print(f"[EX] Extracting {zip_path.name} -> {dest_dir}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"[OK] Extracted to {EXTRACTED_DIR}")


def _verify(directory: Path, expected: list[str]) -> bool:
    """Return True if all expected files exist inside *directory*."""
    return all((directory / f).exists() for f in expected)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: download zip if missing ─────────────────────────────────────
    if ZIP_PATH.exists():
        print(f"[OK] Zip already present at {ZIP_PATH} - skipping download.")
    else:
        _download(URL, ZIP_PATH)

    # ── Step 2: extract if target dir missing or incomplete ──────────────────
    if EXTRACTED_DIR.exists() and _verify(EXTRACTED_DIR, EXPECTED_FILES):
        print(f"[OK] Dataset already extracted at {EXTRACTED_DIR} - skipping extraction.")
    else:
        _extract(ZIP_PATH, RAW_DIR)

    # ── Step 3: final sanity check ───────────────────────────────────────────
    if not _verify(EXTRACTED_DIR, EXPECTED_FILES):
        print(
            "[ERR] Extraction verification failed - some expected files are missing.",
            file=sys.stderr,
        )
        return 1

    print("\n[DONE] MovieLens ml-latest-small is ready.")
    print(f"   Location : {EXTRACTED_DIR}")
    for f in EXPECTED_FILES:
        path = EXTRACTED_DIR / f
        size_kb = path.stat().st_size / 1024
        print(f"   {f:<20} {size_kb:>8.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
