"""
ml/collaborative_model.py
-------------------------
Collaborative filtering recommendation algorithms for CineMatch AI:
1. ItemKNN: item-item mean-centered cosine similarity with top-N neighbors.
2. BiasedMF: explicit-feedback matrix factorization (global + user + item bias + dot product) via SGD.
3. ImplicitALS: implicit-feedback alternating least squares (ratings >= 4.0 as positive interactions).
"""

from __future__ import annotations

from pathlib import Path

import implicit
import numpy as np
import pandas as pd
from scipy import sparse

from ml.base import Recommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class ItemKNN(Recommender):
    """Item-based Collaborative Filtering with mean-centered cosine similarity.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of most similar neighbors retained per item.
    min_similarity : float, default=0.0
        Threshold below which similarity is zeroed out.
    """

    def __init__(self, n_neighbors: int = 20, min_similarity: float = 0.0):
        self.n_neighbors = int(n_neighbors)
        self.min_similarity = float(min_similarity)

        self.n_users: int = 0
        self.n_items: int = 0
        self.global_mean: float = 3.5
        self.user_means: dict[int, float] = {}
        self.seen_by_user: dict[int, set[int]] = {}

        # Sparse neighbor weights matrix W where W[i, j] = sim(i, j) for neighbor i of item j
        self.W: sparse.csr_matrix | None = None
        self.W_abs: sparse.csr_matrix | None = None
        self.R_centered: sparse.csr_matrix | None = None

    def fit(self, train_df: pd.DataFrame) -> ItemKNN:
        """Compute user-centered item-item similarities and construct top-N neighbor graph."""
        user_col = "user_idx"
        item_col = "item_idx"
        rating_col = "rating"

        self.n_users = int(train_df[user_col].max()) + 1
        self.n_items = int(train_df[item_col].max()) + 1
        self.global_mean = float(train_df[rating_col].mean())

        # Compute user means and center ratings
        self.user_means = train_df.groupby(user_col)[rating_col].mean().to_dict()
        self.seen_by_user = train_df.groupby(user_col)[item_col].apply(set).to_dict()

        centered_ratings = train_df[rating_col] - train_df[user_col].map(self.user_means)

        self.R_centered = sparse.csr_matrix(
            (centered_ratings.to_numpy(dtype=np.float32), (train_df[user_col], train_df[item_col])),
            shape=(self.n_users, self.n_items),
        )

        # L2-normalize item columns
        item_norms = np.sqrt(np.asarray(self.R_centered.power(2).sum(axis=0)).ravel())
        item_norms[item_norms == 0] = 1.0

        R_norm = self.R_centered.multiply(1.0 / item_norms).tocsr()

        # Compute item-item cosine similarity matrix S = R_norm.T @ R_norm
        S = (R_norm.T @ R_norm).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()

        # Build top-N neighbor matrices W and |W|
        rows, cols, data = [], [], []

        for j in range(self.n_items):
            start = S.indptr[j]
            end = S.indptr[j + 1]
            if start == end:
                continue

            sims = S.data[start:end]
            nbrs = S.indices[start:end]

            # Retain only positive similarities > min_similarity
            valid_mask = sims > self.min_similarity
            if not np.any(valid_mask):
                continue

            sims = sims[valid_mask]
            nbrs = nbrs[valid_mask]

            if len(sims) > self.n_neighbors:
                top_idx = np.argpartition(sims, -self.n_neighbors)[-self.n_neighbors :]
                sims = sims[top_idx]
                nbrs = nbrs[top_idx]

            for nb, s in zip(nbrs, sims, strict=False):
                rows.append(nb)
                cols.append(j)
                data.append(s)

        self.W = sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32
        )
        self.W_abs = sparse.csr_matrix(
            (np.abs(data), (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32
        )
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for (user_idx, item_idx) using weighted average over rated neighbors."""
        u_mean = self.user_means.get(user_idx, self.global_mean)
        if self.W is None or self.R_centered is None or item_idx >= self.n_items:
            return u_mean

        start = self.W.indptr[item_idx]
        end = self.W.indptr[item_idx + 1]
        if start == end:
            return u_mean

        nbrs = self.W.indices[start:end]
        weights = self.W.data[start:end]

        u_row = self.R_centered[user_idx]
        u_ratings = dict(zip(u_row.indices, u_row.data, strict=False))

        num = 0.0
        denom = 0.0
        for nb, w in zip(nbrs, weights, strict=False):
            if nb in u_ratings:
                num += w * u_ratings[nb]
                denom += abs(w)

        if denom <= 1e-6:
            return u_mean

        return float(np.clip(u_mean + (num / denom), 0.5, 5.0))

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend top-K items using fast sparse matrix-vector dot product."""
        if self.W is None or self.W_abs is None or self.R_centered is None:
            raise ValueError("ItemKNN model must be fitted before calling recommend.")

        u_mean = self.user_means.get(user_idx, self.global_mean)
        u_vec = self.R_centered[user_idx]  # 1 x M

        num = (u_vec @ self.W).toarray().ravel()

        u_binary = sparse.csr_matrix(
            (np.ones(u_vec.nnz, dtype=np.float32), u_vec.indices, [0, u_vec.nnz]),
            shape=(1, self.n_items),
            dtype=np.float32,
        )
        denom = (u_binary @ self.W_abs).toarray().ravel()

        scores = np.where(denom > 0, u_mean + (num / (denom + 1e-6)), u_mean)

        if exclude_seen:
            seen = self.seen_by_user.get(user_idx, set())
            if seen:
                scores[list(seen)] = -9999.0

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


class BiasedMF(Recommender):
    """Explicit-feedback Matrix Factorization with global, user, and item biases trained via SGD.

    Formula:
        r_hat_{ui} = mu + b_u + b_i + p_u . q_i

    Parameters
    ----------
    n_factors : int, default=32
        Latent dimension size d.
    lr : float, default=0.005
        SGD learning rate gamma.
    reg : float, default=0.05
        L2 regularization parameter lambda.
    n_epochs : int, default=15
        Number of training epochs.
    seed : int, default=42
        Random seed for factor initialization.
    """

    def __init__(
        self,
        n_factors: int = 32,
        lr: float = 0.005,
        reg: float = 0.05,
        n_epochs: int = 15,
        seed: int = 42,
    ):
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)

        self.mu: float = 3.5
        self.n_users: int = 0
        self.n_items: int = 0

        self.b_u: np.ndarray | None = None
        self.b_i: np.ndarray | None = None
        self.P: np.ndarray | None = None  # User factors (n_users x d)
        self.Q: np.ndarray | None = None  # Item factors (n_items x d)
        self.seen_by_user: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> BiasedMF:
        """Fit biased matrix factorization model using SGD in NumPy."""
        user_col = "user_idx"
        item_col = "item_idx"
        rating_col = "rating"

        self.n_users = int(train_df[user_col].max()) + 1
        self.n_items = int(train_df[item_col].max()) + 1
        self.mu = float(train_df[rating_col].mean())
        self.seen_by_user = train_df.groupby(user_col)[item_col].apply(set).to_dict()

        u_indices = train_df[user_col].to_numpy(dtype=np.int32)
        i_indices = train_df[item_col].to_numpy(dtype=np.int32)
        ratings = train_df[rating_col].to_numpy(dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        self.b_u = np.zeros(self.n_users, dtype=np.float32)
        self.b_i = np.zeros(self.n_items, dtype=np.float32)
        self.P = rng.normal(0, 0.05, size=(self.n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.05, size=(self.n_items, self.n_factors)).astype(np.float32)

        n_samples = len(ratings)
        order = np.arange(n_samples)

        mu = self.mu
        lr = self.lr
        reg = self.reg
        b_u = self.b_u
        b_i = self.b_i
        P = self.P
        Q = self.Q

        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for idx in order:
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]

                pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
                err = r - pred

                b_u[u] += lr * (err - reg * b_u[u])
                b_i[i] += lr * (err - reg * b_i[i])

                p_u_old = P[u].copy()
                P[u] += lr * (err * Q[i] - reg * P[u])
                Q[i] += lr * (err * p_u_old - reg * Q[i])

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict explicit rating score."""
        if self.b_u is None or self.b_i is None or self.P is None or self.Q is None:
            return self.mu

        bu = self.b_u[user_idx] if user_idx < self.n_users else 0.0
        bi = self.b_i[item_idx] if item_idx < self.n_items else 0.0

        if user_idx < self.n_users and item_idx < self.n_items:
            dot = float(np.dot(self.P[user_idx], self.Q[item_idx]))
        else:
            dot = 0.0

        return float(np.clip(self.mu + bu + bi + dot, 0.5, 5.0))

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend top-K items sorted by predicted rating."""
        if self.b_u is None or self.b_i is None or self.P is None or self.Q is None:
            raise ValueError("BiasedMF model must be fitted before recommend.")

        bu = self.b_u[user_idx] if user_idx < self.n_users else 0.0
        p_u = (
            self.P[user_idx]
            if user_idx < self.n_users
            else np.zeros(self.n_factors, dtype=np.float32)
        )

        scores = self.mu + bu + self.b_i + np.dot(self.Q, p_u)

        if exclude_seen:
            seen = self.seen_by_user.get(user_idx, set())
            if seen:
                scores[list(seen)] = -9999.0

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def save_factors(
        self,
        user_path: Path | str,
        item_path: Path | str,
    ) -> None:
        """Save user and item factor matrices to disk."""
        np.save(str(user_path), self.P)
        np.save(str(item_path), self.Q)


class ImplicitALS(Recommender):
    """Implicit-feedback Matrix Factorization using Alternating Least Squares (ALS).

    Treats ratings >= relevance_threshold as positive implicit interactions.

    Parameters
    ----------
    n_factors : int, default=32
        Number of latent factors.
    regularization : float, default=0.1
        L2 regularization.
    iterations : int, default=15
        Number of ALS optimization iterations.
    alpha : float, default=1.0
        Confidence weighting factor.
    relevance_threshold : float, default=4.0
        Threshold above which ratings are treated as positive preferences.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_factors: int = 32,
        regularization: float = 0.1,
        iterations: int = 15,
        alpha: float = 1.0,
        relevance_threshold: float = 4.0,
        seed: int = 42,
    ):
        self.n_factors = int(n_factors)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.relevance_threshold = float(relevance_threshold)
        self.seed = int(seed)

        self.model: implicit.als.AlternatingLeastSquares | None = None
        self.user_items: sparse.csr_matrix | None = None
        self.seen_by_user: dict[int, set[int]] = {}
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(self, train_df: pd.DataFrame) -> ImplicitALS:
        """Fit Implicit ALS on positive interactions (rating >= relevance_threshold)."""
        user_col = "user_idx"
        item_col = "item_idx"
        rating_col = "rating"

        self.n_users = int(train_df[user_col].max()) + 1
        self.n_items = int(train_df[item_col].max()) + 1
        self.seen_by_user = train_df.groupby(user_col)[item_col].apply(set).to_dict()

        # Binary preference for interactions >= threshold
        pos_df = train_df[train_df[rating_col] >= self.relevance_threshold]

        # Confidence: c = 1 + alpha * (r - threshold + 1) or constant 1.0
        confidences = (
            1.0 + self.alpha * (pos_df[rating_col] - self.relevance_threshold + 1.0)
        ).to_numpy(dtype=np.float32)

        self.user_items = sparse.csr_matrix(
            (confidences, (pos_df[user_col], pos_df[item_col])),
            shape=(self.n_users, self.n_items),
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
        )

        self.model.fit(self.user_items, show_progress=False)
        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate top-K implicit recommendations."""
        if self.model is None or self.user_items is None:
            raise ValueError("ImplicitALS model must be fitted before calling recommend.")

        if user_idx >= self.n_users:
            return []

        u_vec = self.user_items[user_idx]
        ids, scores = self.model.recommend(
            user_idx,
            u_vec,
            N=k + len(self.seen_by_user.get(user_idx, set())),
            filter_already_liked_items=False,
        )

        seen = self.seen_by_user.get(user_idx, set()) if exclude_seen else set()
        recs = []
        for item_id, score in zip(ids, scores, strict=False):
            if item_id not in seen:
                recs.append((int(item_id), float(score)))
                if len(recs) == k:
                    break

        return recs

    def save_factors(
        self,
        user_path: Path | str,
        item_path: Path | str,
    ) -> None:
        """Save factor matrices to disk."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        np.save(str(user_path), self.model.user_factors)
        np.save(str(item_path), self.model.item_factors)
