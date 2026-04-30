# AI-Based Online Review Authenticity Detection

> **A Comparative Study of NLP and Machine Learning Approaches for Fake Review Detection**

---

## 📋 Overview

This project implements a comprehensive AI-based system for detecting fake/deceptive online reviews using Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning techniques. The system analyzes review text to distinguish genuine reviews from fake ones, improving trust and transparency on online platforms.

## 🏗️ Project Structure

```
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Cleaned & preprocessed data
├── src/
│   ├── data_loader.py          # Dataset loading & synthetic generation
│   ├── preprocessing.py        # Text cleaning, tokenization, normalization
│   ├── feature_engineering.py  # TF-IDF, n-grams, linguistic features
│   ├── models/
│   │   ├── classical_ml.py     # Logistic Regression, SVM, Random Forest, XGBoost, NB
│   │   ├── lstm_model.py       # Bidirectional LSTM with attention
│   │   └── bert_model.py       # DistilBERT fine-tuning
│   ├── evaluation.py           # Metrics, confusion matrices, ROC curves
│   ├── statistical_analysis.py # Statistical tests & significance analysis
│   └── visualization.py        # Charts, plots, word clouds
├── results/
│   ├── figures/                # Generated plots & visualizations
│   └── metrics/                # Saved metrics CSV files
├── research_paper/
│   └── paper.md                # Full research paper
├── main.py                     # Main pipeline orchestrator
├── config.py                   # Hyperparameters & configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Run everything (data → training → evaluation → visualization)
python main.py --stage all

# Skip BERT to save time on CPU
python main.py --stage all --skip-bert

# Skip both BERT and LSTM (classical ML only)
python main.py --stage all --skip-bert --skip-lstm
```

### 3. Run Individual Stages

```bash
python main.py --stage data       # Only data generation & preprocessing
python main.py --stage train      # Only model training
python main.py --stage visualize  # Only visualization generation
```
python main.py --stage visualize
### 4. Use Your Own Dataset

```bash
python main.py --stage all --dataset path/to/your/reviews.csv
```

Your CSV should have these columns:
- `review_text`: The review text
- `rating`: Star rating (1-5)
- `label`: 0 for genuine, 1 for fake

## 🔬 Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| Logistic Regression | Classical ML | Linear classifier with L2 regularization |
| Multinomial Naive Bayes | Classical ML | Probabilistic classifier for text |
| Support Vector Machine | Classical ML | SVM with linear and RBF kernels |
| Random Forest | Classical ML | Ensemble of decision trees |
| XGBoost | Classical ML | Gradient boosting classifier |
| BiLSTM | Deep Learning | Bidirectional LSTM with attention mechanism |
| DistilBERT | Deep Learning | Fine-tuned transformer model |

## 📊 Features Extracted

### TF-IDF Features
- Word-level unigrams + bigrams (5000 features)
- Character-level n-grams 3-5 chars (3000 features)

### Linguistic / Stylometric Features
- Average word length, sentence length
- Vocabulary richness (type-token ratio)
- Punctuation density, exclamation ratio
- Capitalization ratio
- First-person pronoun usage
- Sentiment polarity & subjectivity
- ALL-CAPS word ratio
- And more...

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC AUC, Average Precision
- Confusion matrices
- Cross-validation scores
- McNemar's test (pairwise model comparison)
- Wilcoxon signed-rank test
- Cohen's d effect size
- 95% confidence intervals

## 📄 Research Paper

The full research paper is available at `research_paper/paper.md` and includes:
- Comprehensive literature review
- Detailed methodology
- Experimental results
- Statistical significance analysis
- Feature importance analysis

## ⚙️ Configuration

All hyperparameters and settings are centralized in `config.py`. Key settings:

```python
DATASET_CONFIG = {
    "n_samples": 10000,
    "fake_ratio": 0.4,
    "test_size": 0.2,
}

LSTM_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 64,
    "epochs": 15,
}

BERT_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "epochs": 3,
    "learning_rate": 2e-5,
}
```

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- HuggingFace Transformers 4.35+
- NLTK 3.8+
- See `requirements.txt` for the full list

## 📝 License

This project is for academic/research purposes.
