# CineMatch AI: Complete Week 10 Recommender Pipeline Report (README2)

> Comprehensive breakdown of all 14 project sections: Problem definition, data provenance, exploratory analysis, models, evaluation, debugging, and industry extensions.

---

## Table of Contents
1. [Section 1: Problem and Users](#1-problem-and-users)
2. [Section 2: Data, Source & Cleaning](#2-data-source--cleaning)
3. [Section 3: Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Section 4: Baselines](#4-baselines)
5. [Section 5: Content-Based Model](#5-content-based-model)
6. [Section 6: Collaborative Model (ItemKNN)](#6-collaborative-model-itemknn)
7. [Section 7: Matrix Factorization (SVD)](#7-matrix-factorization-svd)
8. [Section 8: Honest Evaluation & Results Table](#8-honest-evaluation--results-table)
9. [Section 9: Cold Start & Hybrid Ensemble](#9-cold-start--hybrid-ensemble)
10. [Section 10: Demo: `recommend()`](#10-demo-recommend)
11. [Section 11: Break-It Robustness Tests](#11-break-it-robustness-tests)
12. [Section 12: By-Hand Mathematical Check](#12-by-hand-mathematical-check)
13. [Section 13: Understanding Pack (Leak, Decision Log, Bug Log)](#13-understanding-pack)
14. [Section 14: AI-Use Log & Industry Scaling](#14-ai-use-log--industry-scaling)

---

## 1. Problem and Users
- **Target User:** Movie streaming platform viewers who face "choice paralysis" when searching through massive catalogs with thousands of titles.
- **What is Recommended:** Highly personalized feature films, cult classics, and niche long-tail titles tailored to user preferences and explicit ratings.
- **Decision Supported:** Deciding *"What should I watch next right now?"* in under 30 seconds.
- **Value Proposition:** Replaces random browsing with transparent, explainable recommendations (*"Why this movie?"*) while preventing over-concentration on mainstream blockbusters.

---

## 2. Data, Source & Cleaning
- **Source & Attribution:** 
  - GroupLens Research at the University of Minnesota: [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/).
  - Enriched with poster images, overviews, and release years via The Movie Database (TMDB) API.
- **Dataset Dimensions:**
  - **Users:** 610 unique users.
  - **Items:** 9,724 rated movies (out of 9,742 total catalog titles).
  - **Ratings:** 100,836 explicit ratings on a 0.5 to 5.0 scale (0.5 increments) with UNIX timestamps.
- **Cleaning & Standardization Steps:**
  1. *Year Extraction:* Regex `r'\((\d{4})\)$'` extracted 4-digit release years from title strings.
  2. *Title Normalization:* Stripped leading/trailing articles and brackets (e.g., `"Shawshank Redemption, The (1994)"` $\rightarrow$ `"The Shawshank Redemption"`, Year: `1994`).
  3. *Genre Splitting:* Pipe-delimited strings (`"Action|Adventure|Sci-Fi"`) parsed into Python sets and multi-hot binary vectors.
  4. *Contiguous Mapping:* Non-contiguous MovieLens IDs mapped to 0-indexed dense matrices `user_idx` $\in [0, 609]$ and `item_idx` $\in [0, 9723]$.
- **Sparsity Calculation:**
  $$\text{Sparsity} = 1.0 - \frac{100,836}{610 \times 9,724} = 1.0 - 0.017006 = 98.30\%$$
  Over 98.3% of the user-item interaction matrix is unobserved (*no opinion*).

---

## 3. Exploratory Data Analysis (EDA)
Two primary phenomena govern the dataset:

### A. Rating Distribution (User Self-Selection Bias)
- Ratings are heavily skewed toward positive feedback: 3.0, 4.0, and 5.0 stars represent **~82% of all ratings**.
- **Mode:** 4.0 stars (over 26,800 ratings). Ratings below 2.0 represent less than 6% of the data.
- **Modelling Implication:** Users selectively choose to watch films they already expect to like. Models must use **user-mean centering** ($r_{ui} - \bar{r}_u$) so that an unrated movie is treated as "neutral/no opinion" rather than a negative zero rating.

### B. Long-Tail Interaction Curve (Popularity Power Law)
- Ranked item frequencies follow a power-law distribution on a log scale:
  - Top 20% of movies account for ~70% of total platform ratings.
  - Over 3,000 movies have 3 or fewer ratings.
- **Modelling Implication:** A pure popularity model will repeatedly push the same 20 blockbusters to every user, causing poor catalog coverage. Collaborative latent factors and content-based features are essential to surface niche titles.

---

## 4. Baselines
Evaluation uses a strict **temporal split** (`split_ratings(df, time_col='time', test_frac=0.20)`). Each user's oldest 80% interactions form the train set; their most recent 20% form the test set.

1. **Global Mean ($\mu$):**
   - Predicts the average rating of the training set ($\mu = 3.5015$) for all test pairs.
   - **Test RMSE:** `1.0422` | **Precision@10:** `0.000` | **NDCG@10:** `0.000`
2. **User Mean ($\bar{r}_u$):**
   - Predicts each user's historical personal mean rating (captures individual generosity/strictness).
   - **Test RMSE:** `0.9416`
3. **Most Popular:**
   - Ranks movies strictly by training rating count.
   - **Test RMSE:** `1.1215` | **Precision@10:** `0.0253` | **Recall@10:** `0.0449` | **NDCG@10:** `0.0431`

---

## 5. Content-Based Model
- **Item Representation:**
  - TF-IDF on user tags (vocabulary size limited to 300, english stopwords removed, fit strictly on item text).
  - Multi-hot binary genre encoding (19 genres).
  - L2 normalized item matrix $\mathbf{M} \in \mathbb{R}^{I \times D}$.
- **User Profile Construction:**
  $$\mathbf{p}_u = \sum_{i \in R_u} (r_{ui} - \bar{r}_u) \mathbf{m}_i$$
  Profiles are normalized by $\|\mathbf{p}_u\|_2$.
- **Scoring & Performance:**
  Cosine similarity $\hat{s}_{ui} = \mathbf{p}_u \cdot \mathbf{m}_i$.
  - **Precision@10:** `0.0023` | **Recall@10:** `0.0021` | **NDCG@10:** `0.0026` | **Catalog Coverage:** `33.81%`

---

## 6. Collaborative Model (ItemKNN)
- **Algorithm:** Item-Item Collaborative Filtering with Mean-Centered Cosine Similarity.
- **Mean-Centering:** Centered user rating vectors $\mathbf{C} = \mathbf{R} - \bar{\mathbf{r}}_u$, with unobserved entries filled as $0$.
- **Similarity Matrix:**
  $$W_{ij} = \frac{\mathbf{c}_i \cdot \mathbf{c}_j}{\|\mathbf{c}_i\|_2 \|\mathbf{c}_j\|_2}$$
- **Top-$k$ Pruning:** For each column, only the top-$k$ ($k=10$) positive nearest neighbors are retained. Self-similarity $W_{ii}$ is set to $0$.
- **Prediction:**
  $$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{j \in N_k(i)} W_{ij}(r_{uj} - \bar{r}_u)}{\sum_{j \in N_k(i)} |W_{ij}| + 10^{-6}}$$
- **Performance:**
  - **RMSE:** `0.9854` | **Precision@10:** `0.0165` | **Recall@10:** `0.0380` | **NDCG@10:** `0.0292`

---

## 7. Matrix Factorization (SVD)
- **Concept:** Decomposes the mean-centered rating matrix into $k$ dense latent taste dimensions:
  $$\hat{\mathbf{R}} = \bar{\mathbf{r}}_u + \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
- **Difference from Neighborhood CF:**
  Neighborhood CF requires explicit co-rating overlap between two specific movies. SVD maps all users and movies into a continuous geometric vector space, allowing it to recommend items that share *no overlapping raters* if their global interaction profiles align.
- **Hyperparameter Sweep:**
  - $k=5$: RMSE = `0.9412`
  - $k=10$: RMSE = `0.9126` (Optimal trade-off)
  - $k=20$: RMSE = `0.9184`
- **SVD ($k=10$) Performance:**
  - **RMSE:** `0.9126` | **Precision@10:** `0.0342` | **Recall@10:** `0.0707` | **NDCG@10:** `0.0611`

---

## 8. Honest Evaluation & Results Table
Evaluated under strict temporal leave-last-20% split on the identical test set:

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Global Mean** | 1.0422 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| **User Mean** | 0.9416 | — | — | — | — |
| **Most Popular** | 1.1215 | 0.0253 | 0.0449 | 0.0431 | 1.0% |
| **Content-Based** | 1.0842 | 0.0023 | 0.0021 | 0.0026 | 33.8% |
| **ItemKNN ($k=10$)** | 0.9854 | 0.0165 | 0.0380 | 0.0292 | 15.9% |
| **SVD ($k=10$)** | 0.9126 | 0.0342 | 0.0707 | 0.0611 | 7.1% |
| **CineMatch Hybrid** | **0.8974** | **0.0320** | **0.0756** | **0.0630** | **10.8%** |

*Winner:* **CineMatch Hybrid Ensemble** wins with the lowest RMSE (0.8974) and highest Recall@10 (0.0756) and NDCG@10 (0.0630).

---

## 9. Cold Start & Hybrid Ensemble
- **Hybrid Fusion Formula:**
  Min-max scales raw predictions to $[0, 1]$ and fuses them via validated weights:
  $$S_{hybrid}(u, i) = 0.60 \cdot \bar{S}_{SVD}(u, i) + 0.25 \cdot \bar{S}_{KNN}(u, i) + 0.15 \cdot \bar{S}_{Content}(u, i)$$
- **Cold Start Handling:**
  - *New User (0 ratings):* Falls back to Bayesian Weighted Rating combined with onboarding genre filters.
  - *New Item (0 ratings):* Evaluated through content TF-IDF & genre embeddings; collaborative models output 0.
  - *Sparse User (1-5 ratings):* ItemKNN neighbors + Content profiling.

---

## 10. Demo: `recommend()`
Interactive recommendation generator:
- Excludes movies the user already rated in the training set.
- Maps internal IDs back to clean titles and genre tags.
- Attaches human-readable explanation reasoning:
  - *"Matches your preference in Sci-Fi"*
  - *"Because you rated The Matrix 5 stars"*

---

## 11. Break-It Robustness Tests
1. **New User (0 Ratings):**
   - *Behavior:* Collaborative models have no history.
   - *Resolution:* Graceful fallback to Bayesian Popularity baseline. Returns classic titles (*Forrest Gump*, *The Shawshank Redemption*).
2. **New Item (0 Ratings):**
   - *Behavior:* Item has no collaborative neighbors.
   - *Resolution:* Content-based cosine similarity matches item metadata against catalog vectors.
3. **Zero Variance (User gives all 5.0 stars):**
   - *Behavior:* $r_{ui} - \bar{r}_u = 0$, leading to zero division in normalization.
   - *Resolution:* Safeguard detects `allclose(weights, 0)` and falls back to raw L1 normalized weights.

---

## 12. By-Hand Mathematical Check
Verifies ItemKNN formula with manual calculation:
- User mean $\bar{r}_u = 4.5$.
- Ratings: Item A ($5.0$), Item B ($4.0$).
- Similarities to target item $T$: $\text{sim}(T, A) = 0.8$, $\text{sim}(T, B) = 0.6$.
- **Manual Formula:**
  $$\hat{r}_{uT} = 4.5 + \frac{0.8 \times (5.0 - 4.5) + 0.6 \times (4.0 - 4.5)}{0.8 + 0.6} = 4.5 + \frac{0.40 - 0.30}{1.40} = 4.5 + 0.0714 = 4.5714$$
- **Code Value:** `4.5714` (Exact match up to floating point precision).

---

## 13. Understanding Pack
### 13a. Deliberate-Leak Experiment
- **Honest Version (Train Only):** User Mean RMSE = `0.9416`
- **Leaky Version (Train + Test):** User Mean RMSE = `0.8841`
- **Insight:** Leaking test interactions artificially deflates error by **0.0575**, proving the necessity of strict temporal boundaries.

### 13b. Decision Log (Key Engineering Choices)
1. *Split Strategy:* Temporal leave-last-20% chosen over random split to prevent future interaction leakage.
2. *Neighborhood Size:* $k=10$ selected over $k=5, 20, 50$ after peak NDCG@10 validation.
3. *Latent Dimensionality:* $k=10$ for TruncatedSVD and $d=64$ for Implicit ALS.
4. *Missing Entries:* Mean-centering chosen over zero-imputation (which caused RMSE to spike to 2.84).
5. *Hybrid Architecture:* 3-way weighted rank fusion chosen over single models for higher catalog coverage.

### 13c. Bug Log (Real Issues Solved)
1. **Empty Tag Vocabulary (`ValueError`):**
   - *Cause:* `TfidfVectorizer` failed when movies had empty tag lists in clean CI environments.
   - *Fix:* Added `if any(text.strip() ...)` check with zero-column CSR matrix fallback.
2. **Argpartition Index Out of Bounds (`ValueError: kth out of bounds`):**
   - *Cause:* Calling `argpartition(scores, -100)` when fewer than 100 candidate items existed.
   - *Fix:* Implemented `actual_k = min(k, len(scores))` safety guard across all models.

---

## 14. AI-Use Log & Industry Scaling
### AI Contributions & Quality Audits
- **FastAPI & Async Architecture:** Generated initial async route scaffolding; manually verified JWT security, password hashing, and CORS settings.
- **CI/CD Debugging:** Diagnosed missing Linux libraries (`greenlet`, `pytest-asyncio`) and fixed sparse matrix normalization edge cases.
- **Error Caught:** AI omitted an empty-vocabulary check in `scikit-learn` normalization; caught via automated GitHub Actions test logs and patched.

### What to Do with 100x More Data (MovieLens 25M / Netflix Prize)
1. **Distributed Two-Tower Vector Retrieval:** Deep user and item neural networks trained with in-batch negative sampling, indexed via **FAISS** or **Google ScaNN** for sub-10ms retrieval.
2. **Distributed Training on Ray / Apache Spark:** Parallelized ALS across GPU cluster nodes.
3. **Multi-Task Ranking:** Neural ranking model predicting both Click-Through Rate (CTR) and Completion Rate.

### How Industry Systems Extend This Project
1. **Two-Stage Architecture (Candidate Retrieval $\rightarrow$ Heavy Ranker):**
   Lightweight ANN filters millions of movies to top 500 candidates; a deep GBDT / DLRM heavy ranker re-scores using real-time contextual features (time of day, device, network speed).
2. **Real-Time Streaming & Feature Store:**
   Kafka events update user interaction history in **Feast / Redis** within milliseconds.
3. **Exploration via Multi-Armed Bandits:**
   Thompson Sampling or Upper Confidence Bound (UCB) algorithms balance exploitation of known tastes with exploration of new content.

