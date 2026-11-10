"""
Machine learning model training and hyperparameter optimization module.

Implements:
- 5-Fold Stratified Cross-Validation strictly on training data.
- Hyperparameter tuning via GridSearchCV over K (1 to 15) and distance metrics.
- Accuracy vs K evaluation curve across K = 1..15.
- 2D KNN model training for decision-boundary projection.
- K-nearest neighbors vector search for interactive neighbor inspection.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import build_pipeline


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_k: int = 15,
    metrics: Tuple[str, ...] = ("euclidean", "manhattan", "minkowski"),
    weights: Tuple[str, ...] = ("uniform", "distance"),
    tune_scaling: bool = True,
    p_values: Tuple[int, ...] = (2, 3),
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Execute exhaustive hyperparameter optimization using 5-Fold Stratified Cross-Validation.

    IMPORTANT: Performed exclusively on the training partition to avoid data leakage.

    Search space:
    - K (n_neighbors): 1 through max_k (default: 1..15)
    - Distance metrics: Euclidean, Manhattan, Minkowski
    - Weights: Uniform, Distance
    - Scaling: StandardScaler vs passthrough (if tune_scaling=True)

    Args:
        X_train: Training features.
        y_train: Training target labels.
        max_k: Maximum neighbor count to search (default: 15).
        metrics: Distance metrics to explore.
        weights: Neighbor weighting schemes.
        tune_scaling: Whether to evaluate StandardScaler vs unscaled passthrough.
        p_values: Values of p to evaluate for Minkowski.
        cv_splits: Number of cross-validation folds (default: 5).
        random_state: Seed for StratifiedKFold shuffling.

    Returns:
        Tuple of (fitted GridSearchCV object, human-readable best parameter dictionary).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    scaler_options = [StandardScaler(), "passthrough"] if tune_scaling else [StandardScaler()]

    param_grid = {
        "scaler": scaler_options,
        "knn__n_neighbors": list(range(1, max_k + 1)),
        "knn__metric": list(metrics),
        "knn__weights": list(weights),
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    best_p = grid_search.best_params_
    scaler_name = "StandardScaler" if isinstance(best_p.get("scaler"), StandardScaler) else "passthrough"

    cleaned_best = {
        "n_neighbors": int(best_p["knn__n_neighbors"]),
        "metric": str(best_p["knn__metric"]),
        "weights": str(best_p["knn__weights"]),
        "scaler": scaler_name,
        "cv_score": float(grid_search.best_score_),
    }

    return grid_search, cleaned_best


def compute_accuracy_vs_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric: str = "euclidean",
    weights: str = "uniform",
    standardize: bool = True,
    p: int = 2,
    max_k: int = 15,
    cv_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute cross-validation and training accuracy for every K from 1 to max_k.

    Strictly evaluated using 5-Fold Stratified Cross-Validation on training data.
    Also calculates training score on the full training set to demonstrate the
    overfitting-underfitting continuum (K=1 overfits with 100% training accuracy).

    Args:
        X_train: Training features.
        y_train: Training labels.
        metric: Distance metric.
        weights: Neighbor weighting.
        standardize: Whether to standardize features.
        p: Minkowski power parameter.
        max_k: Maximum neighbor count (default: 15).
        cv_splits: CV fold count (default: 5).
        random_state: Random state seed.

    Returns:
        pd.DataFrame containing ['k', 'cv_mean', 'cv_std', 'train_acc', 'rank'].
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results: List[Dict[str, Any]] = []

    for k in range(1, max_k + 1):
        pipe = build_pipeline(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            standardize=standardize,
            p=p,
        )

        # 5-fold CV on training data
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        mean_cv = float(np.mean(scores))
        std_cv = float(np.std(scores))

        # Training score on full training set
        pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        train_acc = float(np.mean(train_pred == y_train))

        results.append({
            "k": k,
            "cv_mean": mean_cv,
            "cv_std": std_cv,
            "train_acc": train_acc,
        })

    df = pd.DataFrame(results)
    # Rank by highest cv_mean, breaking ties with lower std dev
    df["rank"] = df.sort_values(by=["cv_mean", "cv_std"], ascending=[False, True]).reset_index().index + 1
    df = df.sort_values(by="k").reset_index(drop=True)
    return df


def train_2d_model(
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    standardize: bool = True,
    p: int = 2,
) -> Pipeline:
    """
    Train an isolated 2D KNN pipeline specifically for decision-boundary projection.

    Args:
        X_train_2d: 2D feature matrix (N, 2).
        y_train: Target labels (N,).
        feature_names: Optional list of 2 feature names.
        n_neighbors: Number of neighbors.
        metric: Distance metric.
        weights: Neighbor weights.
        standardize: Whether to apply StandardScaler.
        p: Minkowski power.

    Returns:
        Fitted 2D Pipeline.
    """
    pipe = build_pipeline(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        standardize=standardize,
        p=p,
    )
    if feature_names is not None and len(feature_names) == 2:
        df_2d = pd.DataFrame(X_train_2d, columns=feature_names)
        pipe.fit(df_2d, y_train)
    else:
        pipe.fit(X_train_2d, y_train)
    return pipe


def get_k_nearest_neighbors(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    input_sample: np.ndarray,
    n_neighbors: int = 5,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Find the exact K nearest training neighbors to an input query sample in feature space.

    Applies the pipeline's fitted StandardScaler transformation to the query sample
    before computing geometric distances.

    Args:
        pipeline: Fitted scikit-learn Pipeline containing StandardScaler and KNeighborsClassifier.
        X_train: Original training feature DataFrame.
        y_train: Original training target Series.
        input_sample: 1D array of feature values for the query sample.
        n_neighbors: Number of nearest neighbors to retrieve.
        class_names: List of human-readable class names.

    Returns:
        DataFrame detailing rank, index, distance, predicted class, and feature values.
    """
    knn: KNeighborsClassifier = pipeline.named_steps["knn"]
    scaler = pipeline.named_steps["scaler"]

    # Wrap in DataFrame with matching column names to maintain feature name consistency
    sample_df = pd.DataFrame(input_sample.reshape(1, -1), columns=X_train.columns)

    # Transform sample if scaler is present
    if scaler != "passthrough" and hasattr(scaler, "transform"):
        sample_transformed = scaler.transform(sample_df)
    else:
        sample_transformed = sample_df.values

    # Actual neighbor search
    actual_k = min(int(n_neighbors), len(X_train))
    distances, indices = knn.kneighbors(sample_transformed, n_neighbors=actual_k)

    neighbor_records = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        target_idx = int(y_train.iloc[idx])
        c_name = class_names[target_idx] if class_names and target_idx < len(class_names) else str(target_idx)
        record = {
            "Rank": rank,
            "Train Sample Index": int(idx),
            "Distance": round(float(dist), 4),
            "Class": c_name,
        }
        # Add key features
        row_feat = X_train.iloc[idx]
        for col in list(X_train.columns)[:4]:  # Top 4 features for readability
            record[str(col)] = round(float(row_feat[col]), 3)

        neighbor_records.append(record)

    return pd.DataFrame(neighbor_records)


def save_model_bytes(model: Any) -> bytes:
    """Serialize model to bytes using joblib."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def load_model_bytes(data: bytes) -> Any:
    """Deserialize model from bytes using joblib."""
    buffer = io.BytesIO(data)
    return joblib.load(buffer)
