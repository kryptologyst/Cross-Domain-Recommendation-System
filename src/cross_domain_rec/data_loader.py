"""Data loading and preprocessing utilities for cross-domain recommendations."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CrossDomainDataLoader:
    """Data loader for cross-domain recommendation datasets."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        random_seed: int = 42,
    ) -> None:
        """Initialize the data loader.

        Args:
            data_dir: Directory containing the data files.
            random_seed: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self._set_seeds()

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load_interactions(
        self, domain: str, filename: str = "interactions.csv"
    ) -> pd.DataFrame:
        """Load interaction data for a specific domain.

        Args:
            domain: Domain name (e.g., 'books', 'movies').
            filename: Name of the interactions file.

        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp.
        """
        file_path = self.data_dir / domain / filename
        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Creating synthetic data.")
            return self._create_synthetic_interactions(domain)

        df = pd.read_csv(file_path)
        required_cols = ["user_id", "item_id", "rating", "timestamp"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        return df

    def load_items(
        self, domain: str, filename: str = "items.csv"
    ) -> pd.DataFrame:
        """Load item metadata for a specific domain.

        Args:
            domain: Domain name (e.g., 'books', 'movies').
            filename: Name of the items file.

        Returns:
            DataFrame with item metadata.
        """
        file_path = self.data_dir / domain / filename
        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Creating synthetic data.")
            return self._create_synthetic_items(domain)

        return pd.read_csv(file_path)

    def load_users(
        self, filename: str = "users.csv"
    ) -> Optional[pd.DataFrame]:
        """Load user metadata.

        Args:
            filename: Name of the users file.

        Returns:
            DataFrame with user metadata or None if file doesn't exist.
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            logger.info(f"User file {file_path} not found. Skipping user metadata.")
            return None

        return pd.read_csv(file_path)

    def _create_synthetic_interactions(self, domain: str) -> pd.DataFrame:
        """Create synthetic interaction data for demonstration.

        Args:
            domain: Domain name.

        Returns:
            Synthetic interaction DataFrame.
        """
        n_users = 1000
        n_items = 500 if domain == "books" else 300

        # Create user-item interactions with some patterns
        interactions = []
        for user_id in range(n_users):
            # Each user interacts with 5-50 items
            n_interactions = np.random.randint(5, 51)
            item_ids = np.random.choice(n_items, n_interactions, replace=False)

            for item_id in item_ids:
                # Rating follows a skewed distribution (more high ratings)
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                timestamp = np.random.randint(1000000000, 1700000000)  # Random timestamp

                interactions.append({
                    "user_id": f"user_{user_id}",
                    "item_id": f"{domain}_{item_id}",
                    "rating": rating,
                    "timestamp": timestamp,
                })

        return pd.DataFrame(interactions)

    def _create_synthetic_items(self, domain: str) -> pd.DataFrame:
        """Create synthetic item metadata for demonstration.

        Args:
            domain: Domain name.

        Returns:
            Synthetic items DataFrame.
        """
        n_items = 500 if domain == "books" else 300

        if domain == "books":
            genres = ["Fiction", "Non-fiction", "Mystery", "Romance", "Sci-Fi", "Fantasy"]
            items = []
            for i in range(n_items):
                # Create numeric features for content-based filtering
                genre_encoded = genres.index(np.random.choice(genres))
                items.append({
                    "item_id": f"books_{i}",
                    "title": f"Book {i}",
                    "genre": np.random.choice(genres),
                    "author": f"Author {i % 50}",
                    "year": np.random.randint(1950, 2024),
                    "genre_encoded": genre_encoded,
                    "length": np.random.randint(100, 800),  # pages
                    "rating_avg": np.random.uniform(3.0, 5.0),
                })
        else:  # movies
            genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]
            items = []
            for i in range(n_items):
                # Create numeric features for content-based filtering
                genre_encoded = genres.index(np.random.choice(genres))
                items.append({
                    "item_id": f"movies_{i}",
                    "title": f"Movie {i}",
                    "genre": np.random.choice(genres),
                    "director": f"Director {i % 30}",
                    "year": np.random.randint(1950, 2024),
                    "genre_encoded": genre_encoded,
                    "duration": np.random.randint(80, 180),  # minutes
                    "rating_avg": np.random.uniform(3.0, 5.0),
                })

        return pd.DataFrame(items)

    def create_train_test_split(
        self,
        interactions: pd.DataFrame,
        test_size: float = 0.2,
        stratify_by_user: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train-test split for interactions.

        Args:
            interactions: Interaction DataFrame.
            test_size: Proportion of data to use for testing.
            stratify_by_user: Whether to stratify by user to ensure each user
                appears in both train and test sets.

        Returns:
            Tuple of (train_interactions, test_interactions).
        """
        if stratify_by_user:
            # Ensure each user appears in both train and test
            train_interactions = []
            test_interactions = []

            for user_id in interactions["user_id"].unique():
                user_interactions = interactions[
                    interactions["user_id"] == user_id
                ].copy()
                if len(user_interactions) < 2:
                    # If user has only one interaction, add to train
                    train_interactions.append(user_interactions)
                else:
                    user_train, user_test = train_test_split(
                        user_interactions,
                        test_size=test_size,
                        random_state=self.random_seed,
                    )
                    train_interactions.append(user_train)
                    test_interactions.append(user_test)

            train_df = pd.concat(train_interactions, ignore_index=True)
            test_df = pd.concat(test_interactions, ignore_index=True)
        else:
            train_df, test_df = train_test_split(
                interactions,
                test_size=test_size,
                random_state=self.random_seed,
            )

        return train_df, test_df

    def create_user_item_matrix(
        self, interactions: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create user-item interaction matrix.

        Args:
            interactions: Interaction DataFrame.

        Returns:
            Tuple of (matrix, user_ids, item_ids).
        """
        # Create pivot table
        matrix = interactions.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )

        user_ids = matrix.index.tolist()
        item_ids = matrix.columns.tolist()
        matrix_values = matrix.values

        return matrix_values, user_ids, item_ids

    def get_negative_samples(
        self,
        interactions: pd.DataFrame,
        n_negatives: int = 1,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate negative samples for implicit feedback.

        Args:
            interactions: Positive interaction DataFrame.
            n_negatives: Number of negative samples per positive interaction.
            random_seed: Random seed for reproducibility.

        Returns:
            DataFrame with negative samples.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Get all unique users and items
        users = interactions["user_id"].unique()
        items = interactions["item_id"].unique()

        # Create set of positive interactions for fast lookup
        positive_interactions = set(
            zip(interactions["user_id"], interactions["item_id"])
        )

        negative_samples = []
        for _, row in interactions.iterrows():
            user_id = row["user_id"]
            for _ in range(n_negatives):
                # Sample random item that user hasn't interacted with
                while True:
                    item_id = np.random.choice(items)
                    if (user_id, item_id) not in positive_interactions:
                        break

                negative_samples.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 0,  # Negative sample
                    "timestamp": row["timestamp"],
                })

        return pd.DataFrame(negative_samples)
