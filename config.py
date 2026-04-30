"""
Configuration file for AI-Based Online Review Authenticity Detection.
Central hub for all hyperparameters, paths, and settings.
"""

import os

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# RANDOM SEED (for reproducibility)
# ============================================================
RANDOM_SEED = 42

# ============================================================
# DATASET CONFIGURATION
# ============================================================
DATASET_CONFIG = {
    "n_samples": 10000,          # Total number of reviews to generate
    "fake_ratio": 0.4,           # 40% fake reviews (realistic distribution)
    "test_size": 0.2,            # 20% test split
    "val_size": 0.1,             # 10% validation split (from train)
}

# ============================================================
# PREPROCESSING
# ============================================================
PREPROCESSING_CONFIG = {
    "min_word_count": 3,         # Minimum words in a review
    "max_word_count": 500,       # Maximum words in a review
    "remove_stopwords": True,
    "lemmatize": True,
    "lowercase": True,
}

# ============================================================
# FEATURE ENGINEERING
# ============================================================
FEATURE_CONFIG = {
    # TF-IDF settings
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),  # Unigrams + Bigrams
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.95,

    # Character-level n-grams
    "char_ngram_range": (3, 5),
    "char_max_features": 3000,

    # Linguistic features to extract
    "linguistic_features": [
        "avg_word_length",
        "avg_sentence_length",
        "vocab_richness",
        "punctuation_density",
        "exclamation_ratio",
        "capital_ratio",
        "first_person_ratio",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "review_length",
        "unique_word_ratio",
    ],
}

# ============================================================
# CLASSICAL ML MODEL HYPERPARAMETERS
# ============================================================
CLASSICAL_ML_CONFIG = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "naive_bayes": {
        "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "xgboost": {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    },
}

# Cross-validation
CV_FOLDS = 5

# ============================================================
# LSTM CONFIGURATION
# ============================================================
LSTM_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "max_sequence_length": 200,
    "vocab_size": 10000,
    "batch_size": 64,
    "epochs": 15,
    "learning_rate": 0.001,
    "patience": 3,           # Early stopping patience
}

# ============================================================
# BERT / DistilBERT CONFIGURATION
# ============================================================
BERT_CONFIG = {
    "model_name": "distilbert-base-uncased",  # CPU-friendly variant
    "max_length": 256,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "weight_decay": 0.01,
}

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================
VIZ_CONFIG = {
    "figure_dpi": 150,
    "figure_format": "png",
    "color_palette": "viridis",
    "font_size": 12,
    "style": "seaborn-v0_8-whitegrid",
}
