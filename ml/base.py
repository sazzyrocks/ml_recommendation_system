"""
ml/base.py
----------
Defines the common base interface for all recommendation models in CineMatch AI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Recommender(ABC):
    """Abstract base class that all recommendation algorithms must implement."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> Recommender:
        """Fit the recommendation model using training interaction data.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame containing training interactions with columns:
            ['user_idx', 'item_idx', 'rating', 'timestamp', ...].

        Returns
        -------
        Recommender
            The fitted instance (self).
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate top-K recommended items for a given user index.

        Parameters
        ----------
        user_idx : int
            Internal 0-based user index.
        k : int, default=10
            Number of recommendations to return.
        exclude_seen : bool, default=True
            Whether to filter out items the user has already rated in train_df.

        Returns
        -------
        list[tuple[int, float]]
            List of (item_idx, predicted_score) tuples, ordered descending by score.
        """
        raise NotImplementedError

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict the explicit rating score for a specific (user_idx, item_idx) pair.

        Optional method for rating prediction models (e.g. SVD, NCF).

        Parameters
        ----------
        user_idx : int
            Internal user index.
        item_idx : int
            Internal item index.

        Returns
        -------
        float
            Predicted rating score.
        """
        raise NotImplementedError("Rating prediction is not implemented for this model.")
