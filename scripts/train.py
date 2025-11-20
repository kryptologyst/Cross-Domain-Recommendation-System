"""Main training and evaluation script for cross-domain recommendation system."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cross_domain_rec.data_loader import CrossDomainDataLoader
from cross_domain_rec.models import (
    SimpleCrossDomainRecommender,
    MatrixFactorizationCrossDomainRecommender,
    ContentBasedCrossDomainRecommender,
)
from cross_domain_rec.evaluation import CrossDomainEvaluator


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("cross_domain_rec.log"),
        ],
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_and_evaluate_models(
    config: Dict,
    data_loader: CrossDomainDataLoader,
    evaluator: CrossDomainEvaluator,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate all models.

    Args:
        config: Configuration dictionary.
        data_loader: Data loader instance.
        evaluator: Evaluator instance.

    Returns:
        Dictionary mapping model names to their evaluation metrics.
    """
    # Load data
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")

    source_interactions = data_loader.load_interactions("books")
    target_interactions = data_loader.load_interactions("movies")
    source_items = data_loader.load_items("books")
    target_items = data_loader.load_items("movies")

    # Create train-test splits
    source_train, source_test = data_loader.create_train_test_split(source_interactions)
    target_train, target_test = data_loader.create_train_test_split(target_interactions)

    logger.info(f"Source domain: {len(source_train)} train, {len(source_test)} test interactions")
    logger.info(f"Target domain: {len(target_train)} train, {len(target_test)} test interactions")

    # Initialize models
    models = {
        "simple_cross_domain": SimpleCrossDomainRecommender(
            random_seed=config["random_seed"]
        ),
        "matrix_factorization": MatrixFactorizationCrossDomainRecommender(
            n_factors=config["mf_n_factors"],
            random_seed=config["random_seed"]
        ),
        "content_based": ContentBasedCrossDomainRecommender(
            random_seed=config["random_seed"]
        ),
    }

    # Train and evaluate models
    results = {}
    k_values = config["evaluation"]["k_values"]

    for model_name, model in tqdm(models.items(), desc="Training models"):
        logger.info(f"Training {model_name}...")

        try:
            # Fit model
            model.fit(
                source_interactions=source_train,
                target_interactions=target_train,
                source_items=source_items,
                target_items=target_items,
            )

            # Evaluate model
            logger.info(f"Evaluating {model_name}...")
            metrics = evaluator.evaluate_model(
                model=model,
                test_interactions=target_test,
                k_values=k_values,
                item_features=target_items,
            )

            results[model_name] = metrics
            logger.info(f"{model_name} evaluation completed")

        except Exception as e:
            logger.error(f"Error training/evaluating {model_name}: {e}")
            continue

    return results


def save_results(results: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Save evaluation results.

    Args:
        results: Evaluation results dictionary.
        output_dir: Output directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_path / "evaluation_results.csv")

    # Create leaderboard
    evaluator = CrossDomainEvaluator()
    leaderboard = evaluator.create_leaderboard(results, primary_metric="ndcg@10")
    leaderboard.to_csv(output_path / "leaderboard.csv", index=False)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(leaderboard.to_string(index=False))


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Cross-domain recommendation system")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found. Using default config.")
        config = {
            "random_seed": 42,
            "mf_n_factors": 50,
            "evaluation": {
                "k_values": [5, 10, 20],
            },
        }

    # Initialize components
    data_loader = CrossDomainDataLoader(
        data_dir=args.data_dir,
        random_seed=config["random_seed"],
    )
    evaluator = CrossDomainEvaluator()

    # Train and evaluate models
    logger.info("Starting model training and evaluation...")
    results = train_and_evaluate_models(config, data_loader, evaluator)

    # Save results
    save_results(results, args.output_dir)

    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()
