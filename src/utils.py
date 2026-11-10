"""
Utility helper functions for SMART ML PARAMETER TUNER.

Includes JSON parameter serialization, CSV export formatting,
and metric styling helpers.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd


def serialize_params(params: Dict[str, Any]) -> str:
    """
    Serialize hyperparameter dictionary into formatted JSON string.

    Args:
        params: Dictionary of parameters.

    Returns:
        Indented JSON string.
    """
    cleaned: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, list)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)

    return json.dumps(cleaned, indent=2)


def export_results_dataframe(
    dataset_name: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    model_type: str,
    k: int,
    metric: str,
    weights: str,
    scaling: str,
    cv_accuracy: float,
    test_accuracy: float,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
) -> pd.DataFrame:
    """
    Construct standardized summary DataFrame for export.

    Returns:
        Single-row summary DataFrame with all experimental parameters and results.
    """
    return pd.DataFrame({
        "Dataset": [dataset_name],
        "Samples": [n_samples],
        "Features": [n_features],
        "Classes": [n_classes],
        "Model Type": [model_type],
        "K (Neighbors)": [k],
        "Distance Metric": [metric],
        "Weights": [weights],
        "Scaling": [scaling],
        "CV Accuracy": [round(cv_accuracy, 4)],
        "Test Accuracy": [round(test_accuracy, 4)],
        "Macro Precision": [round(macro_precision, 4)],
        "Macro Recall": [round(macro_recall, 4)],
        "Macro F1": [round(macro_f1, 4)],
    })
