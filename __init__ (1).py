"""
Evaluation Module
==================
Comprehensive model evaluation with:
- Per-model metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion matrices
- ROC curves (comparative)
- Precision-Recall curves
- Model comparison summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report, average_precision_score
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, METRICS_DIR, VIZ_CONFIG


def setup_plot_style():
    """Configure matplotlib style for publication-quality plots."""
    try:
        plt.style.use(VIZ_CONFIG["style"])
    except Exception:
        plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.size': VIZ_CONFIG["font_size"],
        'figure.dpi': VIZ_CONFIG["figure_dpi"],
        'figure.figsize': (10, 6),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def evaluate_model(y_true, y_pred, y_prob, model_name="Model"):
    """
    Compute comprehensive evaluation metrics for a single model.

    Returns
    -------
    dict
        Dictionary of all metrics.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
    }

    print(f"\n--- {model_name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:20s}: {v:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save=True):
    """Plot a normalized confusion matrix."""
    setup_plot_style()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine', 'Fake'],
                yticklabels=['Genuine', 'Fake'], ax=axes[0])
    axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Oranges',
                xticklabels=['Genuine', 'Fake'],
                yticklabels=['Genuine', 'Fake'], ax=axes[1])
    axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()

    if save:
        safe_name = model_name.lower().replace(' ', '_')
        filepath = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.png")
        plt.savefig(filepath, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches='tight')
        print(f"  Saved: {filepath}")

    plt.show(block=False)
    return cm


def plot_roc_curves(all_results, y_test, save=True):
    """
    Plot ROC curves for all models on a single figure.

    Parameters
    ----------
    all_results : dict
        Dictionary of model_name -> result dict with 'y_prob' key.
    y_test : ndarray
        True test labels.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))

    for (name, result), color in zip(all_results.items(), colors):
        if "error" in result or "y_prob" not in result:
            continue

        y_prob = result["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, "roc_curves_comparison.png")
        plt.savefig(filepath, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.show(block=False)


def plot_precision_recall_curves(all_results, y_test, save=True):
    """Plot Precision-Recall curves for all models."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

    for (name, result), color in zip(all_results.items(), colors):
        if "error" in result or "y_prob" not in result:
            continue

        y_prob = result["y_prob"]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        ax.plot(recall, precision, color=color, linewidth=2,
                label=f'{name} (AP = {ap:.4f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Model Comparison')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, "precision_recall_curves.png")
        plt.savefig(filepath, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.show(block=False)


def plot_cv_scores(all_results, save=True):
    """Plot cross-validation score distributions as box plots."""
    setup_plot_style()

    cv_data = {}
    for name, result in all_results.items():
        if "cv_scores" in result:
            cv_data[name] = result["cv_scores"]

    if not cv_data:
        print("No cross-validation scores available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = range(1, len(cv_data) + 1)
    labels = list(cv_data.keys())
    data = list(cv_data.values())

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Cross-Validation F1 Score Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        filepath = os.path.join(FIGURES_DIR, "cv_scores_boxplot.png")
        plt.savefig(filepath, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.show(block=False)


def generate_comparison_table(all_results, y_test, save=True):
    """
    Generate a comprehensive model comparison table.

    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics.
    """
    rows = []
    for name, result in all_results.items():
        if "error" in result:
            continue

        y_pred = result["y_pred"]
        y_prob = result["y_prob"]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_prob),
            "Avg Precision": average_precision_score(y_test, y_prob),
        }

        if "cv_scores" in result:
            row["CV F1 (mean)"] = result["cv_scores"].mean()
            row["CV F1 (std)"] = result["cv_scores"].std()

        if "metrics" in result and "training_time" in result["metrics"]:
            row["Training Time (s)"] = result["metrics"]["training_time"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    if save:
        filepath = os.path.join(METRICS_DIR, "model_comparison.csv")
        df.to_csv(filepath, index=False)
        print(f"\nModel comparison saved to {filepath}")

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False, float_format="%.4f"))

    return df


def run_full_evaluation(all_results, y_test):
    """
    Run complete evaluation pipeline for all models.

    Parameters
    ----------
    all_results : dict
        Dictionary of model_name -> result dict.
    y_test : ndarray
        True test labels.
    """
    print("\n" + "=" * 60)
    print("RUNNING FULL EVALUATION")
    print("=" * 60)

    # Individual confusion matrices
    for name, result in all_results.items():
        if "error" in result or "y_pred" not in result:
            continue
        plot_confusion_matrix(y_test, result["y_pred"], name)

    # Comparative plots
    plot_roc_curves(all_results, y_test)
    plot_precision_recall_curves(all_results, y_test)
    plot_cv_scores(all_results)

    # Comparison table
    comparison_df = generate_comparison_table(all_results, y_test)

    print("\nEvaluation complete! All figures saved to:", FIGURES_DIR)
    return comparison_df
