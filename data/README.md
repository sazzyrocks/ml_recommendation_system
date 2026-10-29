# Data Directory Documentation

This directory contains the datasets used by CineMatch AI.

## Directory Layout

```
data/
├── raw/
│   ├── ml-latest-small.zip        # Downloaded archive (git-ignored)
│   └── ml-latest-small/           # Extracted raw dataset (git-ignored)
│       ├── movies.csv
│       ├── ratings.csv
│       ├── tags.csv
│       ├── links.csv
│       └── README.txt
├── processed/                     # Cleaned / engineered datasets for modeling (git-ignored)
└── README.md                      # Documentation (tracked)
```

Raw and processed data files are excluded from Git tracking via `.gitignore`. You can re-acquire the raw data at any time using:

```bash
python ml/download_data.py
```

---

## Dataset Overview: MovieLens `ml-latest-small`

- **Source:** GroupLens Research, University of Minnesota (<https://grouplens.org/datasets/movielens/>)
- **Scope:** 100,836 ratings and 3,683 tag applications across 9,742 movies by 610 users.
- **Activity Period:** March 29, 1996 – September 24, 2018 (generated September 26, 2018).
- **User Selection:** Randomly selected users, each having rated at least 20 movies. All user IDs are anonymized.

---

## File Schema & Descriptions

### 1. `movies.csv`
Contains movie metadata and genres.

| Column | Type | Description |
|---|---|---|
| `movieId` | integer | Unique identifier for the movie in MovieLens (`https://movielens.org/movies/<movieId>`). |
| `title` | string | Full movie title, typically including the release year in parentheses (e.g., `Toy Story (1995)`). |
| `genres` | string | Pipe-separated list (`|`) of movie genres (e.g., `Adventure|Animation|Children|Comedy|Fantasy`), or `(no genres listed)`. |

*Available genres:* Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, and `(no genres listed)`.

---

### 2. `ratings.csv`
Contains 5-star rating history across users and movies.

| Column | Type | Description |
|---|---|---|
| `userId` | integer | Anonymized user identifier (consistent across `ratings.csv` and `tags.csv`). |
| `movieId` | integer | Target movie identifier (references `movies.csv`). |
| `rating` | float | Rating score on a 5-star scale with half-star increments (0.5 to 5.0). |
| `timestamp` | integer | Seconds since UTC epoch (January 1, 1970 00:00:00 UTC). |

---

### 3. `tags.csv`
Contains free-form user-applied tag metadata for movies.

| Column | Type | Description |
|---|---|---|
| `userId` | integer | Identifier of user who assigned the tag. |
| `movieId` | integer | Target movie identifier (references `movies.csv`). |
| `tag` | string | User-generated tag word or short phrase (e.g., `funny`, `atmospheric`, `mindfuck`). |
| `timestamp` | integer | Seconds since UTC epoch when tag was applied. |

---

### 4. `links.csv`
Cross-reference mapping identifiers to external movie databases.

| Column | Type | Description |
|---|---|---|
| `movieId` | integer | Identifier in MovieLens (`https://movielens.org/movies/<movieId>`). |
| `imdbId` | string/int | Identifier on IMDb (`http://www.imdb.com/title/tt<imdbId>/` formatted with leading zeros). |
| `tmdbId` | float/int | Identifier on The Movie Database TMDB (`https://www.themoviedb.org/movie/<tmdbId>`). |

---

## License & Usage Terms

The dataset is provided by GroupLens Research at the University of Minnesota under the following conditions:

1. **Non-Commercial Research:** The dataset may be used for research and development purposes. It may not be used for any commercial or revenue-bearing purposes without prior permission from GroupLens.
2. **No Endorsement:** Users may not state or imply endorsement from the University of Minnesota or the GroupLens Research Group.
3. **Redistribution:** Redistribution (including transformations) is permitted only under these same license conditions.
4. **Warranty:** The data and scripts are provided "as is" without warranty of any kind.

### Citation Requirement

Publications or projects using this dataset must cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19.  
> DOI: <https://doi.org/10.1145/2827872>
