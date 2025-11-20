"""Cross-domain recommendation models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class BaseCrossDomainRecommender(ABC):
    """Abstract base class for cross-domain recommenders."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the recommender.

        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    @abstractmethod
    def fit(
        self,
        source_interactions: pd.DataFrame,
        target_interactions: pd.DataFrame,
        source_items: Optional[pd.DataFrame] = None,
        target_items: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fit the model to the data.

        Args:
            source_interactions: Source domain interactions.
            target_interactions: Target domain interactions.
            source_items: Source domain item metadata.
            target_items: Target domain item metadata.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_interacted: bool = True,
    ) -> List[str]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID to recommend for.
            n_recommendations: Number of recommendations to generate.
            exclude_interacted: Whether to exclude items the user has already interacted with.

        Returns:
            List of recommended item IDs.
        """
        pass

    def get_user_similarity(self, user_id: str) -> Dict[str, float]:
        """Get user similarity scores.

        Args:
            user_id: User ID.

        Returns:
            Dictionary mapping user IDs to similarity scores.
        """
        return {}


class SimpleCrossDomainRecommender(BaseCrossDomainRecommender):
    """Simple cross-domain recommender using user preference transfer."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the simple cross-domain recommender."""
        super().__init__(random_seed)
        self.source_matrix: Optional[np.ndarray] = None
        self.target_matrix: Optional[np.ndarray] = None
        self.source_user_ids: Optional[List[str]] = None
        self.target_user_ids: Optional[List[str]] = None
        self.source_item_ids: Optional[List[str]] = None
        self.target_item_ids: Optional[List[str]] = None
        self.user_similarity_matrix: Optional[np.ndarray] = None

    def fit(
        self,
        source_interactions: pd.DataFrame,
        target_interactions: pd.DataFrame,
        source_items: Optional[pd.DataFrame] = None,
        target_items: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fit the simple cross-domain model.

        Args:
            source_interactions: Source domain interactions.
            target_interactions: Target domain interactions.
            source_items: Source domain item metadata (unused).
            target_items: Target domain item metadata (unused).
        """
        # Create user-item matrices
        source_pivot = source_interactions.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        target_pivot = target_interactions.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )

        self.source_matrix = source_pivot.values
        self.target_matrix = target_pivot.values
        self.source_user_ids = source_pivot.index.tolist()
        self.target_user_ids = target_pivot.index.tolist()
        self.source_item_ids = source_pivot.columns.tolist()
        self.target_item_ids = target_pivot.columns.tolist()

        # Compute user similarity in source domain
        self.user_similarity_matrix = cosine_similarity(self.source_matrix)

        logger.info(f"Fitted model with {len(self.source_user_ids)} source users "
                   f"and {len(self.target_user_ids)} target users")

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_interacted: bool = True,
    ) -> List[str]:
        """Generate recommendations using user preference transfer.

        Args:
            user_id: User ID to recommend for.
            n_recommendations: Number of recommendations to generate.
            exclude_interacted: Whether to exclude items the user has already interacted with.

        Returns:
            List of recommended item IDs.
        """
        if self.source_matrix is None or self.target_matrix is None:
            raise ValueError("Model must be fitted before making recommendations")

        # Find user index in source domain
        if user_id not in self.source_user_ids:
            logger.warning(f"User {user_id} not found in source domain")
            return []

        user_idx = self.source_user_ids.index(user_id)

        # Get user's preferences in source domain
        user_preferences = self.source_matrix[user_idx]

        # Find similar users in source domain
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_users_idx = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users

        # Transfer preferences to target domain
        target_scores = np.zeros(len(self.target_item_ids))

        for similar_user_idx in similar_users_idx:
            similarity = user_similarities[similar_user_idx]
            if similarity > 0.1:  # Only consider users with meaningful similarity
                # Find this similar user in target domain
                similar_user_id = self.source_user_ids[similar_user_idx]
                if similar_user_id in self.target_user_ids:
                    target_user_idx = self.target_user_ids.index(similar_user_id)
                    target_scores += similarity * self.target_matrix[target_user_idx]

        # Get items to exclude if needed
        exclude_items = set()
        if exclude_interacted and user_id in self.target_user_ids:
            target_user_idx = self.target_user_ids.index(user_id)
            interacted_items = np.where(self.target_matrix[target_user_idx] > 0)[0]
            exclude_items = {self.target_item_ids[i] for i in interacted_items}

        # Get top recommendations
        top_items_idx = np.argsort(target_scores)[::-1]
        recommendations = []

        for item_idx in top_items_idx:
            item_id = self.target_item_ids[item_idx]
            if item_id not in exclude_items:
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break

        return recommendations

    def get_user_similarity(self, user_id: str) -> Dict[str, float]:
        """Get user similarity scores.

        Args:
            user_id: User ID.

        Returns:
            Dictionary mapping user IDs to similarity scores.
        """
        if self.user_similarity_matrix is None:
            return {}

        if user_id not in self.source_user_ids:
            return {}

        user_idx = self.source_user_ids.index(user_id)
        similarities = self.user_similarity_matrix[user_idx]

        return {
            self.source_user_ids[i]: float(similarities[i])
            for i in range(len(self.source_user_ids))
            if similarities[i] > 0.1
        }


class MatrixFactorizationCrossDomainRecommender(BaseCrossDomainRecommender):
    """Cross-domain recommender using matrix factorization."""

    def __init__(
        self,
        n_factors: int = 50,
        random_seed: int = 42,
    ) -> None:
        """Initialize the matrix factorization cross-domain recommender.

        Args:
            n_factors: Number of latent factors.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(random_seed)
        self.n_factors = n_factors
        self.source_model: Optional[NMF] = None
        self.target_model: Optional[NMF] = None
        self.source_user_factors: Optional[np.ndarray] = None
        self.target_user_factors: Optional[np.ndarray] = None
        self.source_item_factors: Optional[np.ndarray] = None
        self.target_item_factors: Optional[np.ndarray] = None
        self.source_user_ids: Optional[List[str]] = None
        self.target_user_ids: Optional[List[str]] = None
        self.source_item_ids: Optional[List[str]] = None
        self.target_item_ids: Optional[List[str]] = None

    def fit(
        self,
        source_interactions: pd.DataFrame,
        target_interactions: pd.DataFrame,
        source_items: Optional[pd.DataFrame] = None,
        target_items: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fit the matrix factorization models.

        Args:
            source_interactions: Source domain interactions.
            target_interactions: Target domain interactions.
            source_items: Source domain item metadata (unused).
            target_items: Target domain item metadata (unused).
        """
        # Create user-item matrices
        source_pivot = source_interactions.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        target_pivot = target_interactions.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )

        self.source_user_ids = source_pivot.index.tolist()
        self.target_user_ids = target_pivot.index.tolist()
        self.source_item_ids = source_pivot.columns.tolist()
        self.target_item_ids = target_pivot.columns.tolist()

        # Fit NMF models
        self.source_model = NMF(
            n_components=self.n_factors,
            random_state=self.random_seed,
            max_iter=200,
        )
        self.target_model = NMF(
            n_components=self.n_factors,
            random_state=self.random_seed,
            max_iter=200,
        )

        self.source_user_factors = self.source_model.fit_transform(source_pivot.values)
        self.source_item_factors = self.source_model.components_.T

        self.target_user_factors = self.target_model.fit_transform(target_pivot.values)
        self.target_item_factors = self.target_model.components_.T

        logger.info(f"Fitted MF models with {self.n_factors} factors")

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_interacted: bool = True,
    ) -> List[str]:
        """Generate recommendations using matrix factorization.

        Args:
            user_id: User ID to recommend for.
            n_recommendations: Number of recommendations to generate.
            exclude_interacted: Whether to exclude items the user has already interacted with.

        Returns:
            List of recommended item IDs.
        """
        if self.source_user_factors is None or self.target_item_factors is None:
            raise ValueError("Model must be fitted before making recommendations")

        # Find user in source domain
        if user_id not in self.source_user_ids:
            logger.warning(f"User {user_id} not found in source domain")
            return []

        user_idx = self.source_user_ids.index(user_id)
        user_factors = self.source_user_factors[user_idx]

        # Compute scores for all target items
        scores = np.dot(user_factors, self.target_item_factors.T)

        # Get items to exclude if needed
        exclude_items = set()
        if exclude_interacted and user_id in self.target_user_ids:
            target_user_idx = self.target_user_ids.index(user_id)
            target_pivot = pd.DataFrame(
                self.target_model.inverse_transform(
                    self.target_user_factors[target_user_idx:target_user_idx+1]
                ),
                index=[user_id],
                columns=self.target_item_ids,
            )
            interacted_items = target_pivot.loc[user_id][target_pivot.loc[user_id] > 0].index
            exclude_items = set(interacted_items)

        # Get top recommendations
        top_items_idx = np.argsort(scores)[::-1]
        recommendations = []

        for item_idx in top_items_idx:
            item_id = self.target_item_ids[item_idx]
            if item_id not in exclude_items:
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break

        return recommendations


class ContentBasedCrossDomainRecommender(BaseCrossDomainRecommender):
    """Content-based cross-domain recommender using item features."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the content-based cross-domain recommender."""
        super().__init__(random_seed)
        self.item_features: Optional[pd.DataFrame] = None
        self.user_profiles: Optional[Dict[str, np.ndarray]] = None
        self.item_similarity_matrix: Optional[np.ndarray] = None

    def fit(
        self,
        source_interactions: pd.DataFrame,
        target_interactions: pd.DataFrame,
        source_items: Optional[pd.DataFrame] = None,
        target_items: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fit the content-based model.

        Args:
            source_interactions: Source domain interactions.
            target_interactions: Target domain interactions.
            source_items: Source domain item metadata.
            target_items: Target domain item metadata.
        """
        if source_items is None or target_items is None:
            logger.warning("Item metadata not provided. Using simple approach.")
            return

        # Combine item features
        self.item_features = pd.concat([source_items, target_items], ignore_index=True)

        # Create user profiles based on source domain interactions
        self.user_profiles = {}
        
        # Get numeric columns only (exclude item_id and text columns)
        numeric_columns = self.item_features.select_dtypes(include=[np.number]).columns.tolist()
        if "item_id" in numeric_columns:
            numeric_columns.remove("item_id")
        
        for user_id in source_interactions["user_id"].unique():
            user_items = source_interactions[
                source_interactions["user_id"] == user_id
            ]
            user_profile = np.zeros(len(numeric_columns), dtype=float)

            for _, row in user_items.iterrows():
                item_id = row["item_id"]
                rating = float(row["rating"])
                if item_id in self.item_features["item_id"].values:
                    item_features = self.item_features[
                        self.item_features["item_id"] == item_id
                    ][numeric_columns].values.flatten()
                    user_profile += rating * item_features

            # Normalize by number of interactions
            if len(user_items) > 0:
                user_profile /= len(user_items)

            self.user_profiles[user_id] = user_profile

        logger.info(f"Fitted content-based model with {len(self.user_profiles)} user profiles")

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_interacted: bool = True,
    ) -> List[str]:
        """Generate content-based recommendations.

        Args:
            user_id: User ID to recommend for.
            n_recommendations: Number of recommendations to generate.
            exclude_interacted: Whether to exclude items the user has already interacted with.

        Returns:
            List of recommended item IDs.
        """
        if self.user_profiles is None or self.item_features is None:
            raise ValueError("Model must be fitted before making recommendations")

        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found in user profiles")
            return []

        user_profile = self.user_profiles[user_id]

        # Get numeric columns only (exclude item_id and text columns)
        numeric_columns = self.item_features.select_dtypes(include=[np.number]).columns.tolist()
        if "item_id" in numeric_columns:
            numeric_columns.remove("item_id")

        # Compute similarity between user profile and all items
        item_features_matrix = self.item_features[numeric_columns].values
        similarities = cosine_similarity([user_profile], item_features_matrix)[0]

        # Create item-similarity mapping
        item_similarities = list(zip(self.item_features["item_id"], similarities))
        item_similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter out source domain items and get top recommendations
        recommendations = []
        for item_id, similarity in item_similarities:
            if item_id.startswith("movies_"):  # Target domain items
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break

        return recommendations
