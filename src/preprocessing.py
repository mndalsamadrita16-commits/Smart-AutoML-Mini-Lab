"""
Data preprocessing pipeline module for SMART ML PARAMETER TUNER.

Implements strict machine learning methodology:
- Stratified train/test split preserving class distributions.
- Scikit-learn Pipeline encapsulating StandardScaler and KNeighborsClassifier.
- Zero data leakage: scalers are strictly fit on training data during fitting and cross-validation.
"""

from __future__ import annotations

from typing import Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into reproducible stratified train and test partitions.

    The held-out test partition is strictly isolated and never touched
    during hyperparameter optimization or cross-validation.

    Args:
        X: Feature matrix.
        y: Target class labels.
        test_size: Proportion of dataset reserved for testing (default: 0.2 = 20%).
        random_state: Seed for reproducible split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_pipeline(
    n_neighbors: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    standardize: bool = True,
    p: int = 2,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with scaling and KNN classifier.

    Ensures zero data leakage: when used inside cross-validation,
    the StandardScaler is fit exclusively on each training fold.

    Args:
        n_neighbors: Number of nearest neighbors K (default: 5).
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski').
        weights: Neighbor voting weights ('uniform' or 'distance').
        standardize: Whether to standardize features via StandardScaler.
        p: Power parameter for Minkowski metric (p=1: Manhattan, p=2: Euclidean, p>=3: Minkowski).

    Returns:
        Configured scikit-learn Pipeline.
    """
    scaler_step = StandardScaler() if standardize else "passthrough"

    knn_kwargs = {
        "n_neighbors": int(n_neighbors),
        "metric": str(metric),
        "weights": str(weights),
    }

    # Only set p for minkowski to avoid deprecation warnings or confusion
    if metric == "minkowski":
        knn_kwargs["p"] = int(p)
    elif metric == "euclidean":
        knn_kwargs["p"] = 2
    elif metric == "manhattan":
        knn_kwargs["p"] = 1

    knn_step = KNeighborsClassifier(**knn_kwargs)

    return Pipeline([
        ("scaler", scaler_step),
        ("knn", knn_step),
    ])
