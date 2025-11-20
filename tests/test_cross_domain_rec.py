"""Unit tests for cross-domain recommendation system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from cross_domain_rec.data_loader import CrossDomainDataLoader
from cross_domain_rec.models import (
    SimpleCrossDomainRecommender,
    MatrixFactorizationCrossDomainRecommender,
    ContentBasedCrossDomainRecommender,
)
from cross_domain_rec.evaluation import CrossDomainEvaluator


class TestCrossDomainDataLoader:
    """Test cases for CrossDomainDataLoader."""

    def test_init(self):
        """Test DataLoader initialization."""
        loader = CrossDomainDataLoader(data_dir="test_data", random_seed=42)
        assert loader.data_dir == Path("test_data")
        assert loader.random_seed == 42

    def test_create_synthetic_interactions(self):
        """Test synthetic interaction creation."""
        loader = CrossDomainDataLoader(random_seed=42)
        interactions = loader._create_synthetic_interactions("books")
        
        assert isinstance(interactions, pd.DataFrame)
        assert len(interactions) > 0
        assert all(col in interactions.columns for col in ["user_id", "item_id", "rating", "timestamp"])
        assert interactions["rating"].min() >= 1
        assert interactions["rating"].max() <= 5

    def test_create_synthetic_items(self):
        """Test synthetic item creation."""
        loader = CrossDomainDataLoader(random_seed=42)
        items = loader._create_synthetic_items("books")
        
        assert isinstance(items, pd.DataFrame)
        assert len(items) > 0
        assert "item_id" in items.columns
        assert "title" in items.columns
        assert "genre" in items.columns

    def test_create_train_test_split(self):
        """Test train-test split creation."""
        loader = CrossDomainDataLoader(random_seed=42)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            "user_id": ["user1", "user1", "user1", "user2", "user2"],
            "item_id": ["item1", "item2", "item3", "item1", "item4"],
            "rating": [5, 4, 3, 2, 5],
            "timestamp": [1000, 1001, 1002, 1003, 1004],
        })
        
        train, test = loader.create_train_test_split(interactions, test_size=0.2)
        
        assert len(train) + len(test) == len(interactions)
        assert len(train) > 0
        assert len(test) > 0

    def test_create_user_item_matrix(self):
        """Test user-item matrix creation."""
        loader = CrossDomainDataLoader(random_seed=42)
        
        interactions = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["item1", "item2", "item1"],
            "rating": [5, 4, 3],
            "timestamp": [1000, 1001, 1002],
        })
        
        matrix, user_ids, item_ids = loader.create_user_item_matrix(interactions)
        
        assert isinstance(matrix, np.ndarray)
        assert len(user_ids) == 2  # 2 unique users
        assert len(item_ids) == 2  # 2 unique items
        assert matrix.shape == (2, 2)

    def test_get_negative_samples(self):
        """Test negative sample generation."""
        loader = CrossDomainDataLoader(random_seed=42)
        
        interactions = pd.DataFrame({
            "user_id": ["user1", "user1"],
            "item_id": ["item1", "item2"],
            "rating": [5, 4],
            "timestamp": [1000, 1001],
        })
        
        negatives = loader.get_negative_samples(interactions, n_negatives=2)
        
        assert isinstance(negatives, pd.DataFrame)
        assert len(negatives) == 4  # 2 positive * 2 negatives
        assert all(negatives["rating"] == 0)


class TestSimpleCrossDomainRecommender:
    """Test cases for SimpleCrossDomainRecommender."""

    def test_init(self):
        """Test recommender initialization."""
        recommender = SimpleCrossDomainRecommender(random_seed=42)
        assert recommender.random_seed == 42

    def test_fit(self):
        """Test model fitting."""
        recommender = SimpleCrossDomainRecommender(random_seed=42)
        
        source_interactions = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["book1", "book2", "book1"],
            "rating": [5, 4, 3],
            "timestamp": [1000, 1001, 1002],
        })
        
        target_interactions = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "item_id": ["movie1", "movie1"],
            "rating": [4, 5],
            "timestamp": [1000, 1001],
        })
        
        recommender.fit(source_interactions, target_interactions)
        
        assert recommender.source_matrix is not None
        assert recommender.target_matrix is not None
        assert recommender.user_similarity_matrix is not None

    def test_recommend(self):
        """Test recommendation generation."""
        recommender = SimpleCrossDomainRecommender(random_seed=42)
        
        source_interactions = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["book1", "book2", "book1"],
            "rating": [5, 4, 3],
            "timestamp": [1000, 1001, 1002],
        })
        
        target_interactions = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "item_id": ["movie1", "movie1"],
            "rating": [4, 5],
            "timestamp": [1000, 1001],
        })
        
        recommender.fit(source_interactions, target_interactions)
        recommendations = recommender.recommend("user1", n_recommendations=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

    def test_recommend_unknown_user(self):
        """Test recommendation for unknown user."""
        recommender = SimpleCrossDomainRecommender(random_seed=42)
        
        source_interactions = pd.DataFrame({
            "user_id": ["user1"],
            "item_id": ["book1"],
            "rating": [5],
            "timestamp": [1000],
        })
        
        target_interactions = pd.DataFrame({
            "user_id": ["user1"],
            "item_id": ["movie1"],
            "rating": [4],
            "timestamp": [1000],
        })
        
        recommender.fit(source_interactions, target_interactions)
        recommendations = recommender.recommend("unknown_user")
        
        assert recommendations == []


class TestMatrixFactorizationCrossDomainRecommender:
    """Test cases for MatrixFactorizationCrossDomainRecommender."""

    def test_init(self):
        """Test recommender initialization."""
        recommender = MatrixFactorizationCrossDomainRecommender(
            n_factors=20, random_seed=42
        )
        assert recommender.n_factors == 20
        assert recommender.random_seed == 42

    def test_fit(self):
        """Test model fitting."""
        recommender = MatrixFactorizationCrossDomainRecommender(
            n_factors=10, random_seed=42
        )
        
        source_interactions = pd.DataFrame({
            "user_id": ["user1", "user1", "user2"],
            "item_id": ["book1", "book2", "book1"],
            "rating": [5, 4, 3],
            "timestamp": [1000, 1001, 1002],
        })
        
        target_interactions = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "item_id": ["movie1", "movie1"],
            "rating": [4, 5],
            "timestamp": [1000, 1001],
        })
        
        recommender.fit(source_interactions, target_interactions)
        
        assert recommender.source_user_factors is not None
        assert recommender.target_item_factors is not None
        assert recommender.source_user_factors.shape[1] == 10


class TestCrossDomainEvaluator:
    """Test cases for CrossDomainEvaluator."""

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = CrossDomainEvaluator()
        assert evaluator.metrics_history == {}

    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        evaluator = CrossDomainEvaluator()
        
        recommendations = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = ["item1", "item3", "item6"]
        
        precision = evaluator.precision_at_k(recommendations, relevant_items, k=5)
        expected = 2 / 5  # 2 relevant items in top 5
        assert precision == expected

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        evaluator = CrossDomainEvaluator()
        
        recommendations = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = ["item1", "item3", "item6"]
        
        recall = evaluator.recall_at_k(recommendations, relevant_items, k=5)
        expected = 2 / 3  # 2 out of 3 relevant items found
        assert recall == expected

    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        evaluator = CrossDomainEvaluator()
        
        recommendations = ["item1", "item2", "item3"]
        relevant_items = ["item1", "item3"]
        
        ndcg = evaluator.ndcg_at_k(recommendations, relevant_items, k=3)
        assert 0 <= ndcg <= 1

    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        evaluator = CrossDomainEvaluator()
        
        recommendations = ["item1", "item2", "item3"]
        relevant_items = ["item1", "item4"]
        
        hit_rate = evaluator.hit_rate_at_k(recommendations, relevant_items, k=3)
        assert hit_rate == 1.0  # item1 is in recommendations

    def test_coverage(self):
        """Test coverage calculation."""
        evaluator = CrossDomainEvaluator()
        
        all_recommendations = [
            ["item1", "item2"],
            ["item2", "item3"],
            ["item1", "item4"],
        ]
        all_items = ["item1", "item2", "item3", "item4", "item5"]
        
        coverage = evaluator.coverage(all_recommendations, all_items)
        expected = 4 / 5  # 4 out of 5 items covered
        assert coverage == expected

    def test_diversity(self):
        """Test diversity calculation."""
        evaluator = CrossDomainEvaluator()
        
        recommendations = ["item1", "item2", "item3"]
        diversity = evaluator.diversity(recommendations)
        
        assert 0 <= diversity <= 1
        assert diversity == 1.0  # All items are different


if __name__ == "__main__":
    pytest.main([__file__])
