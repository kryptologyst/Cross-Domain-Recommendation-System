#!/usr/bin/env python3
"""Example usage of the cross-domain recommendation system."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cross_domain_rec.data_loader import CrossDomainDataLoader
from cross_domain_rec.models import (
    SimpleCrossDomainRecommender,
    MatrixFactorizationCrossDomainRecommender,
    ContentBasedCrossDomainRecommender,
)
from cross_domain_rec.evaluation import CrossDomainEvaluator


def main():
    """Demonstrate cross-domain recommendation system usage."""
    print("🎯 Cross-Domain Recommendation System Demo")
    print("=" * 50)
    
    # Initialize data loader
    print("\n1. Loading data...")
    data_loader = CrossDomainDataLoader(data_dir="data", random_seed=42)
    
    # Load synthetic data (will be generated if not available)
    source_interactions = data_loader.load_interactions("books")
    target_interactions = data_loader.load_interactions("movies")
    source_items = data_loader.load_items("books")
    target_items = data_loader.load_items("movies")
    
    print(f"   📚 Books: {len(source_interactions)} interactions, {len(source_items)} items")
    print(f"   🎬 Movies: {len(target_interactions)} interactions, {len(target_items)} items")
    
    # Create train-test splits
    print("\n2. Creating train-test splits...")
    source_train, source_test = data_loader.create_train_test_split(source_interactions)
    target_train, target_test = data_loader.create_train_test_split(target_interactions)
    
    print(f"   Training: {len(source_train)} source, {len(target_train)} target interactions")
    print(f"   Testing: {len(source_test)} source, {len(target_test)} target interactions")
    
    # Initialize models
    print("\n3. Initializing models...")
    models = {
        "Simple Cross-Domain": SimpleCrossDomainRecommender(random_seed=42),
        "Matrix Factorization": MatrixFactorizationCrossDomainRecommender(
            n_factors=50, random_seed=42
        ),
        "Content-Based": ContentBasedCrossDomainRecommender(random_seed=42),
    }
    
    # Train models
    print("\n4. Training models...")
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(
            source_interactions=source_train,
            target_interactions=target_train,
            source_items=source_items,
            target_items=target_items,
        )
        print(f"   ✅ {name} trained successfully")
    
    # Evaluate models
    print("\n5. Evaluating models...")
    evaluator = CrossDomainEvaluator()
    results = {}
    
    for name, model in models.items():
        print(f"   Evaluating {name}...")
        try:
            metrics = evaluator.evaluate_model(
                model=model,
                test_interactions=target_test,
                k_values=[5, 10, 20],
                item_features=target_items,
            )
            results[name] = metrics
            print(f"   ✅ {name} evaluated successfully")
        except Exception as e:
            print(f"   ❌ Error evaluating {name}: {e}")
    
    # Display results
    if results:
        print("\n6. Evaluation Results:")
        print("-" * 30)
        
        # Create leaderboard
        leaderboard = evaluator.create_leaderboard(results, primary_metric="ndcg@10")
        print("\nModel Leaderboard (ranked by NDCG@10):")
        print(leaderboard.round(4).to_string(index=False))
        
        # Show detailed metrics
        print("\nDetailed Metrics:")
        results_df = pd.DataFrame(results).T
        key_metrics = ["precision@10", "recall@10", "ndcg@10", "coverage", "diversity"]
        print(results_df[key_metrics].round(4).to_string())
    
    # Generate sample recommendations
    print("\n7. Sample Recommendations:")
    print("-" * 30)
    
    # Get a sample user
    sample_user = source_interactions["user_id"].iloc[0]
    print(f"\nGenerating movie recommendations for user: {sample_user}")
    
    # Show user's book preferences
    user_books = source_interactions[source_interactions["user_id"] == sample_user]
    user_books_with_details = user_books.merge(source_items, on="item_id", how="left")
    
    print("\nUser's book preferences:")
    for _, book in user_books_with_details.iterrows():
        print(f"  📖 {book['title']} ({book['year']}) - {book['genre']} - Rating: {book['rating']}")
    
    # Generate recommendations from all models
    print(f"\nMovie recommendations:")
    for name, model in models.items():
        try:
            recommendations = model.recommend(sample_user, n_recommendations=5)
            print(f"\n{name}:")
            
            for i, item_id in enumerate(recommendations, 1):
                item_info = target_items[target_items["item_id"] == item_id]
                if not item_info.empty:
                    title = item_info.iloc[0]["title"]
                    genre = item_info.iloc[0]["genre"]
                    year = item_info.iloc[0]["year"]
                    print(f"  {i}. 🎬 {title} ({year}) - {genre}")
        except Exception as e:
            print(f"\n{name}: Error - {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo explore more features:")
    print("  • Run the interactive demo: streamlit run scripts/demo.py")
    print("  • Check the analysis notebook: notebooks/analysis.ipynb")
    print("  • Run full training: python scripts/train.py")


if __name__ == "__main__":
    import pandas as pd
    main()
