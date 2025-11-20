"""Evaluation metrics and utilities for cross-domain recommendations."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class CrossDomainEvaluator:
    """Evaluator for cross-domain recommendation models."""

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.metrics_history: Dict[str, List[float]] = {}

    def precision_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int = 10,
    ) -> float:
        """Calculate Precision@K.

        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.

        Returns:
            Precision@K score.
        """
        if k == 0:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        return relevant_in_top_k / k

    def recall_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int = 10,
    ) -> float:
        """Calculate Recall@K.

        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.

        Returns:
            Recall@K score.
        """
        if len(relevant_items) == 0:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        return relevant_in_top_k / len(relevant_items)

    def ndcg_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int = 10,
    ) -> float:
        """Calculate NDCG@K.

        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.

        Returns:
            NDCG@K score.
        """
        if k == 0:
            return 0.0

        top_k_recs = recommendations[:k]
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def map_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int = 10,
    ) -> float:
        """Calculate MAP@K (Mean Average Precision).

        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.

        Returns:
            MAP@K score.
        """
        if len(relevant_items) == 0:
            return 0.0

        top_k_recs = recommendations[:k]
        precision_sum = 0.0
        relevant_count = 0

        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        return precision_sum / len(relevant_items)

    def hit_rate_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int = 10,
    ) -> float:
        """Calculate Hit Rate@K.

        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.

        Returns:
            Hit Rate@K score (1 if any relevant item in top-k, 0 otherwise).
        """
        top_k_recs = recommendations[:k]
        return 1.0 if len(set(top_k_recs) & set(relevant_items)) > 0 else 0.0

    def coverage(
        self,
        all_recommendations: List[List[str]],
        all_items: List[str],
    ) -> float:
        """Calculate catalog coverage.

        Args:
            all_recommendations: List of recommendation lists for all users.
            all_items: List of all available items.

        Returns:
            Coverage score (proportion of items that can be recommended).
        """
        if len(all_items) == 0:
            return 0.0

        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)

        return len(recommended_items) / len(all_items)

    def diversity(
        self,
        recommendations: List[str],
        item_features: Optional[pd.DataFrame] = None,
    ) -> float:
        """Calculate intra-list diversity.

        Args:
            recommendations: List of recommended item IDs.
            item_features: DataFrame with item features for diversity calculation.

        Returns:
            Diversity score (average pairwise dissimilarity).
        """
        if len(recommendations) < 2:
            return 0.0

        if item_features is None:
            # Simple diversity based on item IDs (assumes different IDs = different items)
            return len(set(recommendations)) / len(recommendations)

        # Calculate pairwise cosine dissimilarity
        from sklearn.metrics.pairwise import cosine_similarity

        item_feature_matrix = []
        for item_id in recommendations:
            if item_id in item_features["item_id"].values:
                features = item_features[
                    item_features["item_id"] == item_id
                ].iloc[0, 1:].values  # Exclude item_id column
                item_feature_matrix.append(features)

        if len(item_feature_matrix) < 2:
            return 0.0

        similarity_matrix = cosine_similarity(item_feature_matrix)
        # Convert similarity to dissimilarity
        dissimilarity_matrix = 1 - similarity_matrix
        # Get upper triangle (excluding diagonal)
        upper_triangle = dissimilarity_matrix[np.triu_indices_from(dissimilarity_matrix, k=1)]
        return np.mean(upper_triangle)

    def popularity_bias(
        self,
        recommendations: List[str],
        item_popularity: Dict[str, int],
    ) -> float:
        """Calculate popularity bias in recommendations.

        Args:
            recommendations: List of recommended item IDs.
            item_popularity: Dictionary mapping item IDs to popularity counts.

        Returns:
            Average popularity of recommended items.
        """
        if len(recommendations) == 0:
            return 0.0

        popularities = [item_popularity.get(item_id, 0) for item_id in recommendations]
        return np.mean(popularities)

    def evaluate_model(
        self,
        model,
        test_interactions: pd.DataFrame,
        k_values: List[int] = [5, 10, 20],
        item_features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Evaluate a model on test data.

        Args:
            model: Trained recommendation model.
            test_interactions: Test interaction data.
            k_values: List of k values for evaluation.
            item_features: Item features for diversity calculation.

        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}

        # Get all users in test set
        test_users = test_interactions["user_id"].unique()
        all_recommendations = []
        all_relevant_items = []

        # Calculate item popularity for bias calculation
        item_popularity = test_interactions["item_id"].value_counts().to_dict()

        for user_id in test_users:
            # Get user's relevant items from test set
            user_test_items = test_interactions[
                test_interactions["user_id"] == user_id
            ]["item_id"].tolist()

            # Generate recommendations
            try:
                recommendations = model.recommend(user_id, n_recommendations=max(k_values))
                all_recommendations.append(recommendations)
                all_relevant_items.append(user_test_items)

                # Calculate metrics for different k values
                for k in k_values:
                    metrics[f"precision@{k}"] = metrics.get(f"precision@{k}", 0) + \
                        self.precision_at_k(recommendations, user_test_items, k)
                    metrics[f"recall@{k}"] = metrics.get(f"recall@{k}", 0) + \
                        self.recall_at_k(recommendations, user_test_items, k)
                    metrics[f"ndcg@{k}"] = metrics.get(f"ndcg@{k}", 0) + \
                        self.ndcg_at_k(recommendations, user_test_items, k)
                    metrics[f"map@{k}"] = metrics.get(f"map@{k}", 0) + \
                        self.map_at_k(recommendations, user_test_items, k)
                    metrics[f"hit_rate@{k}"] = metrics.get(f"hit_rate@{k}", 0) + \
                        self.hit_rate_at_k(recommendations, user_test_items, k)

            except Exception as e:
                logger.warning(f"Failed to generate recommendations for user {user_id}: {e}")
                continue

        # Average metrics across users
        n_users = len(test_users)
        for metric_name in metrics:
            metrics[metric_name] /= n_users

        # Calculate additional metrics
        metrics["coverage"] = self.coverage(all_recommendations, list(item_popularity.keys()))

        # Calculate average diversity
        diversities = []
        for recs in all_recommendations:
            if len(recs) > 1:
                diversities.append(self.diversity(recs, item_features))
        metrics["diversity"] = np.mean(diversities) if diversities else 0.0

        # Calculate average popularity bias
        popularity_biases = []
        for recs in all_recommendations:
            if recs:
                popularity_biases.append(self.popularity_bias(recs, item_popularity))
        metrics["popularity_bias"] = np.mean(popularity_biases) if popularity_biases else 0.0

        return metrics

    def create_leaderboard(
        self,
        model_results: Dict[str, Dict[str, float]],
        primary_metric: str = "ndcg@10",
    ) -> pd.DataFrame:
        """Create a model leaderboard.

        Args:
            model_results: Dictionary mapping model names to their evaluation metrics.
            primary_metric: Primary metric for ranking models.

        Returns:
            DataFrame with model rankings.
        """
        leaderboard_data = []
        for model_name, metrics in model_results.items():
            row = {"model": model_name}
            row.update(metrics)
            leaderboard_data.append(row)

        leaderboard = pd.DataFrame(leaderboard_data)
        if primary_metric in leaderboard.columns:
            leaderboard = leaderboard.sort_values(primary_metric, ascending=False)
            leaderboard["rank"] = range(1, len(leaderboard) + 1)

        return leaderboard
