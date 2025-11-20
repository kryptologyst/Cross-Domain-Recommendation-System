# Cross-Domain Recommendation System

A production-ready cross-domain recommendation system that recommends movies based on user preferences in books. This project demonstrates multiple approaches to cross-domain recommendation including user preference transfer, matrix factorization, and content-based filtering.

## Features

- **Multiple Recommendation Models**: Simple cross-domain transfer, matrix factorization, and content-based filtering
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, Coverage, Diversity, and Popularity Bias metrics
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production-Ready**: Type hints, comprehensive documentation, unit tests, and CI/CD pipeline
- **Reproducible**: Deterministic seeding and synthetic data generation for consistent results

## Project Structure

```
cross-domain-recommendation/
├── src/
│   └── cross_domain_rec/
│       ├── __init__.py
│       ├── data_loader.py      # Data loading and preprocessing
│       ├── models.py           # Recommendation models
│       └── evaluation.py       # Evaluation metrics and utilities
├── data/                       # Data directory (auto-generated if empty)
├── models/                     # Saved model artifacts
├── configs/
│   └── default.yaml           # Configuration file
├── notebooks/                  # Jupyter notebooks for analysis
├── scripts/
│   ├── train.py              # Main training script
│   └── demo.py               # Streamlit demo
├── tests/
│   └── test_cross_domain_rec.py
├── assets/                    # Static assets for documentation
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Cross-Domain-Recommendation-System.git
cd Cross-Domain-Recommendation-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Run Quick Demo

```bash
# Run simple demo (recommended for first-time users)
python scripts/simple_example.py

# Run full training and evaluation
python scripts/train.py

# With custom configuration
python scripts/train.py --config configs/default.yaml --data_dir data --output_dir results
```

### 3. Launch Interactive Demo

```bash
# Start Streamlit demo
streamlit run scripts/demo.py
```

The demo will be available at `http://localhost:8501` with the following features:
- **User Recommendations**: Select a user and get movie recommendations based on their book preferences
- **Item Search**: Search for movies and find similar items
- **Model Evaluation**: Compare model performance with interactive plots
- **Data Overview**: Explore dataset statistics and distributions

## Dataset Schema

The system expects the following data structure:

### Interactions Data (`interactions.csv`)
```csv
user_id,item_id,rating,timestamp
user_1,book_1,5,1000000000
user_1,movie_1,4,1000000001
...
```

### Items Data (`items.csv`)
```csv
item_id,title,genre,year,author/director
book_1,"The Great Gatsby",Fiction,1925,F. Scott Fitzgerald
movie_1,"The Great Gatsby",Drama,2013,Baz Luhrmann
...
```

### Users Data (`users.csv`) - Optional
```csv
user_id,age,gender,location
user_1,25,Male,New York
...
```

## Models

### 1. Simple Cross-Domain Recommender
Transfers user preferences from source domain (books) to target domain (movies) using user similarity in the source domain.

**Key Features:**
- User similarity computation using cosine similarity
- Preference transfer based on similar users
- Fast inference suitable for real-time recommendations

### 2. Matrix Factorization Cross-Domain Recommender
Uses Non-negative Matrix Factorization (NMF) to learn latent user and item factors, enabling cross-domain recommendation through shared user representations.

**Key Features:**
- Latent factor learning for both domains
- Configurable number of factors
- Handles sparse interaction data effectively

### 3. Content-Based Cross-Domain Recommender
Leverages item metadata (genre, year, etc.) to create user profiles and recommend items with similar content features.

**Key Features:**
- User profile creation based on interacted items
- Content similarity using item features
- Good for cold-start scenarios

## Evaluation Metrics

The system provides comprehensive evaluation using multiple metrics:

- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Whether any relevant item appears in top-K
- **Coverage**: Proportion of catalog that can be recommended
- **Diversity**: Intra-list diversity of recommendations
- **Popularity Bias**: Average popularity of recommended items

## Configuration

The system uses YAML configuration files. Key parameters:

```yaml
random_seed: 42
mf_n_factors: 50

evaluation:
  k_values: [5, 10, 20]

data:
  source_domain: "books"
  target_domain: "movies"
  test_size: 0.2
  min_interactions_per_user: 5

models:
  simple_cross_domain:
    enabled: true
  matrix_factorization:
    enabled: true
    n_factors: 50
    max_iter: 200
  content_based:
    enabled: true
```

## API Usage

### Basic Usage

```python
from cross_domain_rec.data_loader import CrossDomainDataLoader
from cross_domain_rec.models import SimpleCrossDomainRecommender
from cross_domain_rec.evaluation import CrossDomainEvaluator

# Load data
data_loader = CrossDomainDataLoader(data_dir="data")
source_interactions = data_loader.load_interactions("books")
target_interactions = data_loader.load_interactions("movies")

# Train model
model = SimpleCrossDomainRecommender(random_seed=42)
model.fit(source_interactions, target_interactions)

# Generate recommendations
recommendations = model.recommend("user_1", n_recommendations=10)
print(f"Recommendations: {recommendations}")

# Evaluate model
evaluator = CrossDomainEvaluator()
metrics = evaluator.evaluate_model(model, target_test)
print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
```

### Advanced Usage

```python
# Matrix Factorization with custom parameters
mf_model = MatrixFactorizationCrossDomainRecommender(
    n_factors=100,
    random_seed=42
)

# Content-based with item features
content_model = ContentBasedCrossDomainRecommender(random_seed=42)
source_items = data_loader.load_items("books")
target_items = data_loader.load_items("movies")
content_model.fit(
    source_interactions, 
    target_interactions,
    source_items, 
    target_items
)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/cross_domain_rec

# Run specific test file
pytest tests/test_cross_domain_rec.py
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/

# Lint code
ruff check src/ tests/ scripts/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Synthetic Data Generation

If no real data is available, the system automatically generates synthetic data with realistic patterns:

- **User-Item Interactions**: Skewed rating distribution (more high ratings)
- **Item Metadata**: Realistic genres, years, and other features
- **Cross-Domain Patterns**: Some correlation between user preferences across domains

## Performance Considerations

- **Memory Usage**: Matrix factorization models require more memory for large datasets
- **Training Time**: Content-based models are fastest to train, MF models take longer
- **Inference Speed**: Simple cross-domain model provides fastest recommendations
- **Scalability**: All models can handle datasets with thousands of users and items

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've installed the package in development mode (`pip install -e .`)
2. **Data Not Found**: The system will generate synthetic data if real data is not available
3. **Memory Issues**: Reduce `mf_n_factors` or use smaller datasets
4. **Slow Training**: Consider using fewer factors or smaller datasets for experimentation

### Logging

The system uses Python's logging module. Set log level in configuration:

```yaml
logging:
  level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
  file: "cross_domain_rec.log"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cross_domain_recommendation,
  title={Cross-Domain Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Cross-Domain-Recommendation-System}
}
```

## Acknowledgments

- Built with modern Python libraries: pandas, scikit-learn, numpy
- Visualization powered by Plotly and Streamlit
- Evaluation metrics based on standard recommendation system literature
- Inspired by cross-domain recommendation research
# Cross-Domain-Recommendation-System
