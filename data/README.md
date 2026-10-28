# data/README.md
# Data Directory

## Layout
| Path | Contents |
|------|----------|
| `raw/` | Original, immutable source files (git-ignored). |
| `processed/` | Cleaned / feature-engineered files ready for modelling (git-ignored). |

## Data sources
- **MovieLens ml-latest-small** — 100 k ratings, ~9 k movies (for development).
  Download: <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>
- **TMDB API** — movie overviews and poster URLs fetched via `ml/preprocessing/fetch_tmdb.py` (not yet written).

## How to populate `raw/`
```powershell
# From the project root
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip" `
    -OutFile "data/raw/ml-latest-small.zip"
Expand-Archive -Path "data/raw/ml-latest-small.zip" -DestinationPath "data/raw/"
```
