"""Streamlit demo for cross-domain recommendation system."""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cross_domain_rec.data_loader import CrossDomainDataLoader
from cross_domain_rec.models import (
    SimpleCrossDomainRecommender,
    MatrixFactorizationCrossDomainRecommender,
    ContentBasedCrossDomainRecommender,
)
from cross_domain_rec.evaluation import CrossDomainEvaluator


@st.cache_data
def load_data_and_models():
    """Load data and train models (cached)."""
    data_loader = CrossDomainDataLoader(data_dir="data", random_seed=42)
    
    # Load data
    source_interactions = data_loader.load_interactions("books")
    target_interactions = data_loader.load_interactions("movies")
    source_items = data_loader.load_items("books")
    target_items = data_loader.load_items("movies")
    
    # Create train-test splits
    source_train, _ = data_loader.create_train_test_split(source_interactions)
    target_train, _ = data_loader.create_train_test_split(target_interactions)
    
    # Initialize and train models
    models = {
        "Simple Cross-Domain": SimpleCrossDomainRecommender(random_seed=42),
        "Matrix Factorization": MatrixFactorizationCrossDomainRecommender(
            n_factors=50, random_seed=42
        ),
        "Content-Based": ContentBasedCrossDomainRecommender(random_seed=42),
    }
    
    for model in models.values():
        model.fit(
            source_interactions=source_train,
            target_interactions=target_train,
            source_items=source_items,
            target_items=target_items,
        )
    
    return {
        "models": models,
        "source_interactions": source_interactions,
        "target_interactions": target_interactions,
        "source_items": source_items,
        "target_items": target_items,
    }


def get_user_recommendations(
    user_id: str,
    model,
    target_items: pd.DataFrame,
    n_recommendations: int = 10,
) -> pd.DataFrame:
    """Get recommendations for a user with explanations."""
    recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
    
    if not recommendations:
        return pd.DataFrame()
    
    # Create recommendations DataFrame with item details
    rec_data = []
    for i, item_id in enumerate(recommendations):
        item_info = target_items[target_items["item_id"] == item_id]
        if not item_info.empty:
            rec_data.append({
                "rank": i + 1,
                "item_id": item_id,
                "title": item_info.iloc[0]["title"],
                "genre": item_info.iloc[0]["genre"],
                "year": item_info.iloc[0]["year"],
            })
    
    return pd.DataFrame(rec_data)


def get_similar_items(
    item_id: str,
    target_items: pd.DataFrame,
    n_similar: int = 5,
) -> pd.DataFrame:
    """Get similar items based on metadata."""
    if target_items.empty:
        return pd.DataFrame()
    
    target_item = target_items[target_items["item_id"] == item_id]
    if target_item.empty:
        return pd.DataFrame()
    
    item_genre = target_item.iloc[0]["genre"]
    item_year = target_item.iloc[0]["year"]
    
    # Find items with same genre and similar year
    similar_items = target_items[
        (target_items["genre"] == item_genre) &
        (target_items["item_id"] != item_id)
    ].copy()
    
    # Sort by year similarity
    similar_items["year_diff"] = abs(similar_items["year"] - item_year)
    similar_items = similar_items.sort_values("year_diff").head(n_similar)
    
    return similar_items[["item_id", "title", "genre", "year"]]


def create_evaluation_plots(results: Dict[str, Dict[str, float]]) -> None:
    """Create evaluation plots."""
    if not results:
        st.warning("No evaluation results available.")
        return
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Precision@K", "Recall@K", "NDCG@K", "Hit Rate@K"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    k_values = [5, 10, 20]
    metrics = ["precision", "recall", "ndcg", "hit_rate"]
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for model_name in results_df.index:
            values = [results_df.loc[model_name, f"{metric}@{k}"] for k in k_values]
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=values,
                    mode="lines+markers",
                    name=model_name,
                    showlegend=(i == 0),  # Only show legend for first subplot
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=600,
        title_text="Model Performance Comparison",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Cross-Domain Recommendation System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Cross-Domain Recommendation System")
    st.markdown("""
    This demo showcases a cross-domain recommendation system that recommends movies 
    based on user preferences in books. The system uses multiple approaches including 
    user preference transfer, matrix factorization, and content-based filtering.
    """)
    
    # Load data and models
    with st.spinner("Loading data and training models..."):
        data = load_data_and_models()
    
    models = data["models"]
    source_interactions = data["source_interactions"]
    target_interactions = data["target_interactions"]
    source_items = data["source_items"]
    target_items = data["target_items"]
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["User Recommendations", "Item Search", "Model Evaluation", "Data Overview"]
    )
    
    if page == "User Recommendations":
        st.header("📚➡️🎬 User Recommendations")
        
        # User selection
        available_users = source_interactions["user_id"].unique()
        selected_user = st.selectbox(
            "Select a user:",
            available_users,
            help="Choose a user to see movie recommendations based on their book preferences"
        )
        
        # Model selection
        selected_model_name = st.selectbox(
            "Select recommendation model:",
            list(models.keys()),
            help="Choose which model to use for recommendations"
        )
        
        selected_model = models[selected_model_name]
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
        
        # Generate recommendations
        if st.button("Generate Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations_df = get_user_recommendations(
                    selected_user, selected_model, target_items, n_recs
                )
                
                if not recommendations_df.empty:
                    st.success(f"Recommendations for {selected_user}:")
                    
                    # Display recommendations table
                    st.dataframe(
                        recommendations_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Show user's book preferences
                    st.subheader("User's Book Preferences")
                    user_books = source_interactions[
                        source_interactions["user_id"] == selected_user
                    ]
                    user_books_with_details = user_books.merge(
                        source_items, on="item_id", how="left"
                    )
                    
                    if not user_books_with_details.empty:
                        st.dataframe(
                            user_books_with_details[["title", "genre", "rating"]],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No book preferences found for this user.")
                else:
                    st.warning("No recommendations available for this user.")
    
    elif page == "Item Search":
        st.header("🔍 Item Search")
        
        # Item search
        search_term = st.text_input(
            "Search for movies:",
            placeholder="Enter movie title or genre..."
        )
        
        if search_term:
            # Filter movies based on search term
            filtered_movies = target_items[
                (target_items["title"].str.contains(search_term, case=False, na=False)) |
                (target_items["genre"].str.contains(search_term, case=False, na=False))
            ]
            
            if not filtered_movies.empty:
                st.subheader(f"Found {len(filtered_movies)} movies:")
                
                for _, movie in filtered_movies.iterrows():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{movie['title']}** ({movie['year']})")
                        st.write(f"Genre: {movie['genre']}")
                    
                    with col2:
                        if st.button(f"Find Similar", key=f"similar_{movie['item_id']}"):
                            similar_items = get_similar_items(
                                movie["item_id"], target_items, n_similar=5
                            )
                            
                            if not similar_items.empty:
                                st.subheader("Similar Movies:")
                                st.dataframe(
                                    similar_items,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("No similar movies found.")
            else:
                st.info("No movies found matching your search.")
    
    elif page == "Model Evaluation":
        st.header("📊 Model Evaluation")
        
        # Run evaluation
        if st.button("Run Model Evaluation"):
            with st.spinner("Evaluating models..."):
                evaluator = CrossDomainEvaluator()
                
                # Create test split for evaluation
                data_loader = CrossDomainDataLoader(data_dir="data", random_seed=42)
                _, target_test = data_loader.create_train_test_split(target_interactions)
                
                results = {}
                for model_name, model in models.items():
                    try:
                        metrics = evaluator.evaluate_model(
                            model=model,
                            test_interactions=target_test,
                            k_values=[5, 10, 20],
                            item_features=target_items,
                        )
                        results[model_name] = metrics
                    except Exception as e:
                        st.error(f"Error evaluating {model_name}: {e}")
                
                if results:
                    # Display results table
                    st.subheader("Evaluation Results")
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Create plots
                    st.subheader("Performance Comparison")
                    create_evaluation_plots(results)
                    
                    # Leaderboard
                    st.subheader("Model Leaderboard")
                    leaderboard = evaluator.create_leaderboard(results, primary_metric="ndcg@10")
                    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
                else:
                    st.error("No evaluation results available.")
    
    elif page == "Data Overview":
        st.header("📈 Data Overview")
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Source Users", len(source_interactions["user_id"].unique()))
        with col2:
            st.metric("Source Items", len(source_interactions["item_id"].unique()))
        with col3:
            st.metric("Target Users", len(target_interactions["user_id"].unique()))
        with col4:
            st.metric("Target Items", len(target_interactions["item_id"].unique()))
        
        # Interaction distribution
        st.subheader("Rating Distribution")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Books", "Movies")
        )
        
        source_ratings = source_interactions["rating"].value_counts().sort_index()
        target_ratings = target_interactions["rating"].value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(x=source_ratings.index, y=source_ratings.values, name="Books"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=target_ratings.index, y=target_ratings.values, name="Movies"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Genre distribution
        st.subheader("Genre Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Books Genres**")
            book_genres = source_items["genre"].value_counts()
            fig_books = px.pie(
                values=book_genres.values,
                names=book_genres.index,
                title="Book Genres"
            )
            st.plotly_chart(fig_books, use_container_width=True)
        
        with col2:
            st.write("**Movie Genres**")
            movie_genres = target_items["genre"].value_counts()
            fig_movies = px.pie(
                values=movie_genres.values,
                names=movie_genres.index,
                title="Movie Genres"
            )
            st.plotly_chart(fig_movies, use_container_width=True)


if __name__ == "__main__":
    main()
