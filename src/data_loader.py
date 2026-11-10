"""
Data loader module for SMART ML PARAMETER TUNER.

Loads real benchmark datasets strictly from scikit-learn:
- Iris
- Wine
- Breast Cancer Wisconsin
- Digits

Provides structured DatasetBundle with features, targets, names, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)


@dataclass
class DatasetBundle:
    """Container holding loaded machine learning dataset and associated metadata."""

    name: str
    description: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    class_names: List[str]
    raw_images: Optional[np.ndarray] = None


def load_dataset(name: str) -> DatasetBundle:
    """
    Load an authentic classification dataset from scikit-learn.

    Supported datasets:
    - 'Iris': Fisher's Iris dataset (150 samples, 4 features, 3 classes)
    - 'Wine': Chemical analysis of Italian wines (178 samples, 13 features, 3 classes)
    - 'Breast Cancer': Diagnostic Wisconsin Breast Cancer (569 samples, 30 features, 2 classes)
    - 'Digits': Handwritten digits 0-9 (1,797 samples, 64 features, 10 classes)

    Args:
        name: Name of the dataset to load.

    Returns:
        DatasetBundle containing X, y, metadata, and class/feature names.

    Raises:
        ValueError: If dataset name is not supported.
    """
    clean_name = name.strip().lower()

    if "iris" in clean_name:
        raw = load_iris(as_frame=True)
        desc = (
            "Fisher's classic Iris botanical dataset containing 150 instances across 3 flower species "
            "(Setosa, Versicolor, Virginica) measured across 4 sepal and petal morphological attributes."
        )
        return DatasetBundle(
            name="Iris",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
        )

    elif "wine" in clean_name:
        raw = load_wine(as_frame=True)
        desc = (
            "Results of a chemical analysis of 178 wines grown in the same region in Italy but derived "
            "from three different cultivars. 13 continuous chemical constituents (alcohol, malic acid, flavonoids, etc.)."
        )
        return DatasetBundle(
            name="Wine",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
        )

    elif "cancer" in clean_name or "breast" in clean_name:
        raw = load_breast_cancer(as_frame=True)
        desc = (
            "Diagnostic Wisconsin Breast Cancer dataset with 569 patient instances and 30 continuous "
            "cell-nucleus features computed from digitized FNA biopsy images. Binary classification: Malignant vs Benign."
        )
        return DatasetBundle(
            name="Breast Cancer",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
        )

    elif "digit" in clean_name:
        raw = load_digits(as_frame=True)
        desc = (
            "Optical recognition of handwritten digits (0 through 9) from 8x8 pixel normalized bitmaps. "
            "1,797 samples across 64 grayscale pixel features and 10 digit classes."
        )
        # raw.images has shape (1797, 8, 8)
        raw_images = getattr(raw, "images", None)
        return DatasetBundle(
            name="Digits",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
            raw_images=raw_images,
        )

    else:
        raise ValueError(
            f"Unsupported dataset '{name}'. Available datasets: 'Iris', 'Wine', 'Breast Cancer', 'Digits'."
        )


def get_dataset_summary(bundle: DatasetBundle) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics.

    Args:
        bundle: Loaded DatasetBundle.

    Returns:
        Dictionary containing sample count, feature count, class count, class distribution,
        and missing value information.
    """
    class_counts = bundle.y.value_counts().sort_index()
    distribution = {
        bundle.class_names[idx]: int(count)
        for idx, count in class_counts.items()
    }

    return {
        "name": bundle.name,
        "n_samples": int(bundle.X.shape[0]),
        "n_features": int(bundle.X.shape[1]),
        "n_classes": len(bundle.class_names),
        "class_distribution": distribution,
        "feature_names": bundle.feature_names,
        "class_names": bundle.class_names,
        "missing_values": int(bundle.X.isna().sum().sum()),
    }
