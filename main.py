"""
Main Pipeline Orchestrator
============================
AI-Based Online Review Authenticity Detection

Orchestrates the full ML pipeline:
1. Data generation / loading
2. Text preprocessing
3. Feature engineering
4. Model training (Classical ML + LSTM + BERT)
5. Evaluation & comparison
6. Statistical analysis
7. Visualization generation

Usage:
    python main.py --stage all        # Run everything
    python main.py --stage data       # Only data preparation
    python main.py --stage train      # Only model training
    python main.py --stage evaluate   # Only evaluation
    python main.py --stage visualize  # Only visualizations
    python main.py --skip-bert        # Skip BERT (saves time on CPU)
    python main.py --skip-lstm        # Skip LSTM
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASET_CONFIG, RANDOM_SEED, PROCESSED_DATA_DIR,
    FIGURES_DIR, METRICS_DIR, RAW_DATA_DIR
)
from src.data_loader import generate_synthetic_dataset, load_dataset
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import (
    build_feature_matrix, extract_linguistic_features
)
from src.models.classical_ml import train_classical_models
from src.evaluation import run_full_evaluation
from src.statistical_analysis import run_statistical_analysis
from src.visualization import generate_all_visualizations


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Based Online Review Authenticity Detection Pipeline"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "data", "train", "evaluate", "visualize"],
        help="Pipeline stage to run (default: all)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to an external CSV dataset (optional)"
    )
    parser.add_argument(
        "--skip-bert", action="store_true",
        help="Skip BERT fine-tuning (saves time on CPU)"
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM training"
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of synthetic samples to generate"
    )
    return parser.parse_args()


def stage_data(args):
    """Stage 1: Data preparation."""
    print("\n" + "#" * 60)
    print("#  STAGE 1: DATA PREPARATION")
    print("#" * 60)

    # Load or generate data
    if args.dataset:
        df = load_dataset(args.dataset)
    else:
        n_samples = args.n_samples or DATASET_CONFIG["n_samples"]
        df = generate_synthetic_dataset(n_samples=n_samples)

    # Preprocess
    df = preprocess_dataframe(df)

    print(f"\nDataset ready: {len(df)} reviews")
    print(f"  Genuine: {sum(df['label'] == 0)}")
    print(f"  Fake: {sum(df['label'] == 1)}")

    print("\n" + "-" * 60)
    print("SAMPLE REVIEW TEXTS:")
    print("-" * 60)
    genuine_sample = df[df['label'] == 0].iloc[0]['review_text']
    fake_sample = df[df['label'] == 1].iloc[0]['review_text']
    print(f"[GENUINE]: {genuine_sample[:150]}...")
    print(f"\n[FAKE]:    {fake_sample[:150]}...")
    print("-" * 60 + "\n")

    return df


def stage_train(df, args):
    """Stage 2: Feature engineering + Model training."""
    print("\n" + "#" * 60)
    print("#  STAGE 2: FEATURE ENGINEERING & MODEL TRAINING")
    print("#" * 60)

    # Train/test split
    df_train, df_test = train_test_split(
        df, test_size=DATASET_CONFIG["test_size"],
        random_state=RANDOM_SEED, stratify=df["label"]
    )

    print(f"\nTrain set: {len(df_train)} | Test set: {len(df_test)}")

    # Build feature matrix
    feature_data = build_feature_matrix(df_train, df_test)

    # -- Classical ML Models --
    classical_results = train_classical_models(
        feature_data["X_train"], feature_data["y_train"],
        feature_data["X_test"], feature_data["y_test"],
    )

    all_results = dict(classical_results)

    # -- LSTM Model --
    if not args.skip_lstm:
        try:
            from src.models.lstm_model import train_lstm_model

            # Use light-processed text for LSTM
            text_col = "light_processed_text" if "light_processed_text" in df_train.columns else "cleaned_text"

            lstm_results = train_lstm_model(
                df_train[text_col], feature_data["y_train"],
                df_test[text_col], feature_data["y_test"],
            )
            all_results["BiLSTM"] = lstm_results
        except Exception as e:
            print(f"\nLSTM training failed: {e}")
            all_results["BiLSTM"] = {"error": str(e)}
    else:
        print("\nSkipping LSTM (--skip-lstm flag)")

    # -- BERT Model --
    if not args.skip_bert:
        try:
            from src.models.bert_model import train_bert_model

            text_col = "cleaned_text" if "cleaned_text" in df_train.columns else "review_text"

            bert_results = train_bert_model(
                df_train[text_col], feature_data["y_train"],
                df_test[text_col], feature_data["y_test"],
            )
            all_results["DistilBERT"] = bert_results
        except Exception as e:
            print(f"\nBERT training failed: {e}")
            all_results["DistilBERT"] = {"error": str(e)}
    else:
        print("\nSkipping BERT (--skip-bert flag)")

    return all_results, feature_data, df_train, df_test


def stage_evaluate(all_results, feature_data, df_train, df_test):
    """Stage 3: Evaluation and analysis."""
    print("\n" + "#" * 60)
    print("#  STAGE 3: EVALUATION & ANALYSIS")
    print("#" * 60)

    y_test = feature_data["y_test"]

    # Full evaluation (confusion matrices, ROC curves, etc.)
    comparison_df = run_full_evaluation(all_results, y_test)

    # Statistical analysis
    stat_data = {
        "X_train_ling": feature_data["X_train_ling"],
        "y_train": feature_data["y_train"],
        "ling_feature_names": feature_data["ling_feature_names"],
    }
    stat_results = run_statistical_analysis(all_results, y_test, stat_data)

    return comparison_df, stat_results


def stage_visualize(df, feature_data, all_results):
    """Stage 4: Generate visualizations."""
    print("\n" + "#" * 60)
    print("#  STAGE 4: VISUALIZATION")
    print("#" * 60)

    # Extract linguistic features for visualization
    ling_df = extract_linguistic_features(df, "review_text")

    # Deep learning results for training curves
    dl_results = {}
    for name in ["BiLSTM", "DistilBERT"]:
        if name in all_results and "error" not in all_results[name]:
            dl_results[name] = all_results[name]

    generate_all_visualizations(
        df=df,
        ling_features_df=ling_df,
        labels=df["label"].values,
        X_features=feature_data.get("X_train") if feature_data else None,
        y_features=feature_data.get("y_train") if feature_data else None,
        dl_results=dl_results if dl_results else None,
    )


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("  AI-BASED ONLINE REVIEW AUTHENTICITY DETECTION")
    print("  Fake Review Detection Pipeline")
    print("=" * 60)
    print(f"  Stage:      {args.stage}")
    print(f"  Skip BERT:  {args.skip_bert}")
    print(f"  Skip LSTM:  {args.skip_lstm}")
    print(f"  Dataset:    {args.dataset or 'Synthetic'}")
    print("=" * 60)

    start_time = time.time()

    if args.stage == "all":
        # Run full pipeline
        df = stage_data(args)
        all_results, feature_data, df_train, df_test = stage_train(df, args)
        comparison_df, stat_results = stage_evaluate(
            all_results, feature_data, df_train, df_test
        )
        stage_visualize(df, feature_data, all_results)

    elif args.stage == "data":
        df = stage_data(args)

    elif args.stage == "train":
        # Load preprocessed data
        filepath = os.path.join(PROCESSED_DATA_DIR, "preprocessed_reviews.csv")
        if not os.path.exists(filepath):
            df = stage_data(args)
        else:
            df = pd.read_csv(filepath)
            print(f"Loaded preprocessed data: {len(df)} reviews")

        all_results, feature_data, df_train, df_test = stage_train(df, args)

    elif args.stage == "evaluate":
        print("Evaluation requires a trained model. Run 'python main.py --stage all' first.")

    elif args.stage == "visualize":
        filepath = os.path.join(PROCESSED_DATA_DIR, "preprocessed_reviews.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"Loaded preprocessed data: {len(df)} reviews")
            ling_df = extract_linguistic_features(df, "review_text")
            generate_all_visualizations(df=df, ling_features_df=ling_df,
                                         labels=df["label"].values)
        else:
            print("No preprocessed data found. Run data stage first.")

    total_time = time.time() - start_time
    
    # Print a highly visible final summary so it doesn't get scrolled off the terminal
    try:
        print("\n" + "*" * 60)
        print("  FINAL SUMMARY: ACCURACY & SAMPLE REVIEWS")
        print("*" * 60)
        
        # Print accuracy matrix if it exists
        metrics_file = os.path.join(METRICS_DIR, "model_comparison.csv")
        if os.path.exists(metrics_file):
            comp_df = pd.read_csv(metrics_file)
            print("\n>>> MODEL ACCURACIES <<<")
            print(comp_df[['Model', 'Accuracy', 'F1 Score']].to_string(index=False))
            
        # Print review text samples if they exist
        data_file = os.path.join(PROCESSED_DATA_DIR, "preprocessed_reviews.csv")
        if os.path.exists(data_file):
            data_df = pd.read_csv(data_file)
            print("\n>>> SAMPLE REVIEWS <<<")
            gen_sample = data_df[data_df['label'] == 0].iloc[0]['review_text']
            fake_sample = data_df[data_df['label'] == 1].iloc[0]['review_text']
            print(f"[GENUINE]: {gen_sample[:200]}...")
            print()
            print(f"[FAKE]:    {fake_sample[:200]}...")
        print("*" * 60)
    except Exception as e:
        import traceback
        print(f"Error printing final summary: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE -- Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 60)

    # Show all non-blocking plots generated during the pipeline
    try:
        import matplotlib.pyplot as plt
        if plt.get_fignums():  # If there are any open figures
            print("\nDisplaying generated plots. Close the plot windows to exit the program.")
            plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
