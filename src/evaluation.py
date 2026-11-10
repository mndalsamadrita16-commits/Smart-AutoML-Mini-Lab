"""
Evaluation module for SMART ML PARAMETER TUNER.

Calculates authentic classification performance metrics dynamically using scikit-learn:
- Accuracy, Precision (macro), Recall (macro), F1-Score (macro).
- Confusion matrix and per-class classification reports.
- Before vs. After optimization comparison on identical held-out test sets.
- Multi-dataset benchmark evaluation across Iris, Wine, Breast Cancer, and Digits.
- Standardization comparison experiment for empirical research inquiry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.preprocessing import build_pipeline, split_data


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit pipeline on training data and evaluate comprehensively on held-out test set.

    Also executes 5-fold Stratified Cross-Validation on training data to provide
    training validation score alongside generalization test score.

    Args:
        pipeline: scikit-learn Pipeline with scaler and KNN classifier.
        X_train: Training features.
        y_train: Training labels.
        X_test: Held-out test features.
        y_test: Held-out test labels.
        class_names: Target class label strings.
        cv_folds: Number of CV folds on training data.
        random_state: Random state seed.

    Returns:
        Comprehensive evaluation dictionary containing metrics, confusion matrix, and report.
    """
    # 1. Cross-validation on training partition only
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # 2. Fit pipeline on full training partition
    pipeline.fit(X_train, y_train)

    # 3. Predict strictly on held-out test partition
    y_pred = pipeline.predict(X_test)

    # 4. Probabilities if supported
    try:
        y_proba = pipeline.predict_proba(X_test)
    except Exception:
        y_proba = None

    # 5. Core test metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # 6. Confusion matrix
    unique_labels = sorted(list(set(y_train).union(set(y_test))))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # 7. Classification report DataFrame
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    return {
        "cv_scores": cv_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "test_metrics": {
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
        },
        "confusion_matrix": cm,
        "report_dict": report_dict,
        "report_df": report_df,
        "fitted_pipeline": pipeline,
    }


def run_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate standard default baseline KNN (K=5, Euclidean, uniform weights, unstandardized).

    Args:
        X_train, y_train: Training partition.
        X_test, y_test: Held-out test partition.
        class_names: Class labels.
        random_state: Random state seed.

    Returns:
        Evaluation dictionary for default baseline model.
    """
    baseline_pipe = build_pipeline(
        n_neighbors=5,
        metric="euclidean",
        weights="uniform",
        standardize=False,
    )
    return evaluate_pipeline(
        baseline_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        cv_folds=5,
        random_state=random_state,
    )


def compare_baseline_vs_tuned(
    default_eval: Dict[str, Any],
    tuned_eval: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute differences between baseline and tuned model evaluated on identical test sets.

    Returns:
        Dictionary containing absolute and percentage differences.
    """
    def_acc = default_eval["test_metrics"]["accuracy"]
    tune_acc = tuned_eval["test_metrics"]["accuracy"]
    def_f1 = default_eval["test_metrics"]["macro_f1"]
    tune_f1 = tuned_eval["test_metrics"]["macro_f1"]

    acc_diff = tune_acc - def_acc
    f1_diff = tune_f1 - def_f1

    return {
        "baseline_accuracy": def_acc,
        "tuned_accuracy": tune_acc,
        "accuracy_diff": acc_diff,
        "accuracy_diff_pct": acc_diff * 100.0,
        "baseline_f1": def_f1,
        "tuned_f1": tune_f1,
        "f1_diff": f1_diff,
    }


def run_multi_dataset_benchmark(
    test_size: float = 0.2,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run authentic end-to-end ML training and hyperparameter search across all 4 datasets:
    Iris, Wine, Breast Cancer, and Digits.

    Every dataset is properly split, scaled within CV, optimized over K=1..15 and distance metrics,
    and evaluated on its own held-out test set.

    Returns:
        List of result records containing dataset specifications and actual scores.
    """
    from src.model import run_grid_search

    dataset_names = ["Iris", "Wine", "Breast Cancer", "Digits"]
    benchmark_records: List[Dict[str, Any]] = []

    for name in dataset_names:
        bundle = load_dataset(name)
        X_tr, X_te, y_tr, y_te = split_data(bundle.X, bundle.y, test_size=test_size, random_state=random_state)

        grid, best_p = run_grid_search(
            X_tr,
            y_tr,
            max_k=15,
            metrics=("euclidean", "manhattan", "minkowski"),
            weights=("uniform", "distance"),
            tune_scaling=True,
            random_state=random_state,
        )

        best_pipe = grid.best_estimator_
        eval_res = evaluate_pipeline(
            best_pipe,
            X_tr,
            y_tr,
            X_te,
            y_te,
            bundle.class_names,
            cv_folds=5,
            random_state=random_state,
        )

        test_acc = eval_res["test_metrics"]["accuracy"]
        test_f1 = eval_res["test_metrics"]["macro_f1"]

        benchmark_records.append({
            "Dataset": bundle.name,
            "Samples": int(bundle.X.shape[0]),
            "Features": int(bundle.X.shape[1]),
            "Classes": len(bundle.class_names),
            "Best K": best_p["n_neighbors"],
            "Best Distance Metric": best_p["metric"].capitalize(),
            "Best Weights": best_p["weights"].capitalize(),
            "Best Scaling": best_p["scaler"],
            "CV Accuracy": round(best_p["cv_score"], 4),
            "Test Accuracy": round(test_acc, 4),
            "Macro F1": round(test_f1, 4),
        })

    return benchmark_records


def run_standardization_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
    k: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Conduct empirical test comparing Unstandardized vs Standardized features
    to answer Research Question 2 with live computed evidence.

    Returns:
        Dictionary containing evaluations for both unscaled and scaled pipelines.
    """
    unscaled_pipe = build_pipeline(
        n_neighbors=k,
        metric=metric,
        weights=weights,
        standardize=False,
    )
    scaled_pipe = build_pipeline(
        n_neighbors=k,
        metric=metric,
        weights=weights,
        standardize=True,
    )

    eval_unscaled = evaluate_pipeline(
        unscaled_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        cv_folds=5,
        random_state=random_state,
    )
    eval_scaled = evaluate_pipeline(
        scaled_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        cv_folds=5,
        random_state=random_state,
    )

    return {
        "unscaled": eval_unscaled,
        "scaled": eval_scaled,
    }
