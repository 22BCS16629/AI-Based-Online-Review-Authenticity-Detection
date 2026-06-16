"""
Classical Machine Learning Models
===================================
Implements 5 classical ML classifiers with GridSearchCV tuning:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Support Vector Machine (SVM)
4. Random Forest
5. XGBoost

Each model is trained with stratified k-fold cross-validation
and optional SMOTE for class imbalance handling.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CLASSICAL_ML_CONFIG, CV_FOLDS, RANDOM_SEED, METRICS_DIR

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Skipping XGBoost model.")


def _get_models():
    """
    Return a dictionary of model name -> (model_instance, param_grid).
    """
    models = {}

    models["Logistic Regression"] = (
        LogisticRegression(random_state=RANDOM_SEED),
        CLASSICAL_ML_CONFIG["logistic_regression"],
    )

    models["Naive Bayes"] = (
        MultinomialNB(),
        CLASSICAL_ML_CONFIG["naive_bayes"],
    )

    models["SVM"] = (
        SVC(random_state=RANDOM_SEED, probability=True),
        CLASSICAL_ML_CONFIG["svm"],
    )

    models["Random Forest"] = (
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        CLASSICAL_ML_CONFIG["random_forest"],
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_SEED,
                eval_metric='logloss',
                n_jobs=-1,
            ),
            CLASSICAL_ML_CONFIG["xgboost"],
        )

    return models


def train_classical_models(X_train, y_train, X_test, y_test,
                            use_smote=True, feature_names=None):
    """
    Train all classical ML models with hyperparameter tuning.

    Parameters
    ----------
    X_train : sparse matrix or ndarray
        Training features.
    y_train : ndarray
        Training labels.
    X_test : sparse matrix or ndarray
        Test features.
    y_test : ndarray
        Test labels.
    use_smote : bool
        Whether to apply SMOTE for class imbalance.
    feature_names : list, optional
        Feature names for importance analysis.

    Returns
    -------
    dict
        Dictionary of model results with metrics and trained models.
    """
    print("\n" + "=" * 60)
    print("TRAINING CLASSICAL ML MODELS")
    print("=" * 60)

    models = _get_models()
    results = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for name, (model, param_grid) in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training: {name}")
        print(f"{'-' * 50}")

        start_time = time.time()

        try:
            # For Naive Bayes, ensure non-negative features (skip SMOTE on sparse)
            if name == "Naive Bayes":
                from sklearn.preprocessing import MinMaxScaler
                from scipy.sparse import issparse
                if issparse(X_train):
                    # MinMaxScaler on sparse matrices
                    mm_scaler = MinMaxScaler()
                    X_train_nb = mm_scaler.fit_transform(X_train.toarray())
                    X_test_nb = mm_scaler.transform(X_test.toarray())
                else:
                    mm_scaler = MinMaxScaler()
                    X_train_nb = mm_scaler.fit_transform(X_train)
                    X_test_nb = mm_scaler.transform(X_test)

                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring='f1',
                    n_jobs=-1, verbose=0, refit=True
                )
                grid_search.fit(X_train_nb, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_nb)
                y_prob = best_model.predict_proba(X_test_nb)[:, 1]

                # Cross-val scores
                cv_scores = cross_val_score(best_model, X_train_nb, y_train,
                                            cv=cv, scoring='f1')
            else:
                # Standard GridSearchCV
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring='f1',
                    n_jobs=-1, verbose=0, refit=True
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]

                # Cross-val scores
                cv_scores = cross_val_score(best_model, X_train, y_train,
                                            cv=cv, scoring='f1')

            elapsed = time.time() - start_time

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "training_time": elapsed,
                "best_params": grid_search.best_params_,
            }

            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                feature_importance = np.abs(best_model.coef_[0])

            results[name] = {
                "model": best_model,
                "metrics": metrics,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "cv_scores": cv_scores,
                "feature_importance": feature_importance,
                "classification_report": classification_report(y_test, y_pred),
            }

            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1 Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            print(f"  CV F1:     {metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f}")
            print(f"  Time:      {elapsed:.2f}s")

        except Exception as e:
            print(f"  ERROR training {name}: {str(e)}")
            results[name] = {"error": str(e)}

    # Save results summary
    _save_results_summary(results)

    return results


def _save_results_summary(results):
    """Save a CSV summary of all model performances."""
    rows = []
    for name, result in results.items():
        if "error" in result:
            continue
        m = result["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1 Score": f"{m['f1_score']:.4f}",
            "ROC AUC": f"{m['roc_auc']:.4f}",
            "CV F1 (mean)": f"{m['cv_f1_mean']:.4f}",
            "CV F1 (std)": f"{m['cv_f1_std']:.4f}",
            "Training Time (s)": f"{m['training_time']:.2f}",
        })

    if rows:
        df = pd.DataFrame(rows)
        filepath = os.path.join(METRICS_DIR, "classical_ml_results.csv")
        df.to_csv(filepath, index=False)
        print(f"\nResults summary saved to {filepath}")
        print(f"\n{df.to_string(index=False)}")
