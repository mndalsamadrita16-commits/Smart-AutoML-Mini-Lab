"""
SMART ML PARAMETER TUNER — AutoML Mini Lab
Package initialization.
"""

from src.data_loader import DatasetBundle, load_dataset, get_dataset_summary
from src.preprocessing import split_data, build_pipeline
from src.model import (
    run_grid_search,
    compute_accuracy_vs_k,
    train_2d_model,
    get_k_nearest_neighbors,
)
from src.evaluation import (
    evaluate_pipeline,
    compare_baseline_vs_tuned,
    run_multi_dataset_benchmark,
    run_standardization_experiment,
)
from src.visualization import (
    plot_accuracy_vs_k,
    plot_decision_boundary,
    plot_confusion_matrix,
    plot_before_after,
    plot_dataset_comparison,
    plot_prediction_probabilities,
    plot_class_distribution,
    plot_training_vs_cv,
    plot_digit_preview,
)
from src.utils import serialize_params, export_results_dataframe

__all__ = [
    "DatasetBundle",
    "load_dataset",
    "get_dataset_summary",
    "split_data",
    "build_pipeline",
    "run_grid_search",
    "compute_accuracy_vs_k",
    "train_2d_model",
    "get_k_nearest_neighbors",
    "evaluate_pipeline",
    "compare_baseline_vs_tuned",
    "run_multi_dataset_benchmark",
    "run_standardization_experiment",
    "plot_accuracy_vs_k",
    "plot_decision_boundary",
    "plot_confusion_matrix",
    "plot_before_after",
    "plot_dataset_comparison",
    "plot_prediction_probabilities",
    "plot_class_distribution",
    "plot_training_vs_cv",
    "plot_digit_preview",
    "serialize_params",
    "export_results_dataframe",
]
