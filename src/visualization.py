"""
Visualization module for SMART ML PARAMETER TUNER.

Constructs interactive Plotly charts:
1. Accuracy vs K validation curve with confidence band and best K annotation.
2. 2D KNN Decision Boundary with mesh contour and train/test scatter overlays.
3. Interactive Confusion Matrix heatmap with count and normalized percentages.
4. Before vs. After Optimization comparison bar chart.
5. Multi-Dataset Benchmark comparison charts.
6. Real-time Prediction class probabilities.
7. Class distribution chart.
8. Training vs. CV curve demonstrating overfitting vs. underfitting.
9. 8x8 pixel digit preview for the Digits dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Premium Color Palette
PRIMARY_TEAL = "#1B8A93"
DARK_NAVY = "#1B3A57"
ACCENT_ORANGE = "#E0A040"
LIGHT_BG = "#F8FAFC"
BORDER_GRAY = "#E2E8F0"
TEXT_COLOR = "#0F172A"

# Class Colors for discrete classification
CLASS_COLORS = [
    "#1B8A93",  # Teal
    "#E0A040",  # Amber/Orange
    "#2563EB",  # Royal Blue
    "#10B981",  # Emerald Green
    "#EF4444",  # Crimson Red
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan
    "#F59E0B",  # Yellow Amber
    "#EC4899",  # Pink
    "#64748B",  # Slate
]


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent styling across all Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "y": 0.96,
            "x": 0.04,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": DARK_NAVY, "family": "Inter, sans-serif"},
        },
        paper_bgcolor="white",
        plot_bgcolor="#FAFCFE",
        font={"family": "Inter, sans-serif", "color": TEXT_COLOR, "size": 12},
        margin={"l": 45, "r": 35, "t": 60, "b": 45},
        hoverlabel={"bgcolor": DARK_NAVY, "font_color": "white", "font_size": 12},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#EDF2F7",
        linecolor="#CBD5E1",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EDF2F7",
        linecolor="#CBD5E1",
        zeroline=False,
    )
    return fig


def plot_accuracy_vs_k(df_k: pd.DataFrame, best_k: Optional[int] = None) -> go.Figure:
    """
    Visualization 1: Accuracy vs K.

    Plots mean 5-fold Stratified Cross-Validation score for every K from 1 to 15
    with a confidence interval standard deviation band and a highlight marker for the best K.

    Args:
        df_k: DataFrame containing ['k', 'cv_mean', 'cv_std', 'train_acc', 'rank'].
        best_k: Best K value discovered via optimization or CV maximum.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Upper and Lower SD band
    upper_band = (df_k["cv_mean"] + df_k["cv_std"]).clip(upper=1.0)
    lower_band = (df_k["cv_mean"] - df_k["cv_std"]).clip(lower=0.0)

    # Shaded SD region
    fig.add_trace(
        go.Scatter(
            x=list(df_k["k"]) + list(df_k["k"][::-1]),
            y=list(upper_band * 100) + list(lower_band[::-1] * 100),
            fill="toself",
            fillcolor="rgba(27, 138, 147, 0.12)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            showlegend=True,
            name="±1 SD Band",
        )
    )

    # Main CV line
    fig.add_trace(
        go.Scatter(
            x=df_k["k"],
            y=df_k["cv_mean"] * 100,
            mode="lines+markers",
            name="CV Accuracy",
            line={"color": PRIMARY_TEAL, "width": 3},
            marker={"size": 8, "color": PRIMARY_TEAL, "symbol": "circle"},
            customdata=np.stack((df_k["cv_std"] * 100, df_k["rank"]), axis=-1),
            hovertemplate=(
                "<b>K = %{x}</b><br>"
                "CV Accuracy: <b>%{y:.2f}%</b><br>"
                "Std Dev: ±%{customdata[0]:.2f}%<br>"
                "CV Rank: #%{customdata[1]}<extra></extra>"
            ),
        )
    )

    # Highlight Best K marker
    if best_k is None:
        best_k = int(df_k.loc[df_k["cv_mean"].idxmax()]["k"])

    best_row = df_k[df_k["k"] == best_k]
    if not best_row.empty:
        best_acc = float(best_row["cv_mean"].iloc[0]) * 100
        fig.add_trace(
            go.Scatter(
                x=[best_k],
                y=[best_acc],
                mode="markers+text",
                name=f"Best K ({best_k})",
                marker={"size": 15, "color": ACCENT_ORANGE, "symbol": "star", "line": {"color": DARK_NAVY, "width": 1.5}},
                text=[f"Best K = {best_k}"],
                textposition="top center",
                textfont={"color": DARK_NAVY, "size": 13, "family": "Inter, sans-serif"},
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        xaxis={"title": "<b>Number of Neighbors (K)</b>", "dtick": 1, "range": [0.5, 15.5]},
        yaxis={"title": "<b>Cross-Validation Accuracy (%)</b>", "ticksuffix": "%"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return _apply_theme(fig, "Accuracy vs K (5-Fold Stratified Cross-Validation)")


def plot_decision_boundary(
    model_2d: Any,
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    X_test_2d: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    resolution: int = 150,
) -> go.Figure:
    """
    Visualization 2: KNN Decision Boundary.

    Builds a 2D meshgrid across feature min-max ranges, queries model_2d.predict(),
    renders colored classification regions, and overlays train & held-out test points.

    Args:
        model_2d: Fitted KNN pipeline trained on 2 selected features.
        X_train_2d: 2D training features.
        y_train: Training labels.
        X_test_2d: 2D test features.
        y_test: Test labels.
        feature_names: Names of [Feature X, Feature Y].
        class_names: Target class label names.
        resolution: Grid resolution.

    Returns:
        Plotly Figure.
    """
    all_x = np.concatenate([X_train_2d[:, 0], X_test_2d[:, 0]])
    all_y = np.concatenate([X_train_2d[:, 1], X_test_2d[:, 1]])

    pad_x = (all_x.max() - all_x.min()) * 0.08 or 0.5
    pad_y = (all_y.max() - all_y.min()) * 0.08 or 0.5

    x_min, x_max = all_x.min() - pad_x, all_x.max() + pad_x
    y_min, y_max = all_y.min() - pad_y, all_y.max() + pad_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=feature_names) if feature_names and len(feature_names) == 2 else grid_points
    zz = model_2d.predict(grid_df).reshape(xx.shape)

    fig = go.Figure()

    # Discrete colorscale for contour regions
    n_classes = len(class_names)
    colorscale = []
    for i, c_idx in enumerate(range(n_classes)):
        color = CLASS_COLORS[c_idx % len(CLASS_COLORS)]
        start = i / n_classes
        end = (i + 1) / n_classes
        colorscale.extend([[start, color], [end, color]])

    # 1. Decision Regions
    fig.add_trace(
        go.Heatmap(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=zz,
            colorscale=colorscale,
            opacity=0.25,
            showscale=False,
            hoverinfo="none",
        )
    )

    # 2. Training Data Points
    for c_idx, c_name in enumerate(class_names):
        mask_tr = y_train == c_idx
        if np.any(mask_tr):
            fig.add_trace(
                go.Scatter(
                    x=X_train_2d[mask_tr, 0],
                    y=X_train_2d[mask_tr, 1],
                    mode="markers",
                    name=f"{c_name} (Train)",
                    marker={
                        "size": 8,
                        "color": CLASS_COLORS[c_idx % len(CLASS_COLORS)],
                        "symbol": "circle",
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate=(
                        f"<b>{c_name} (Train)</b><br>"
                        f"{feature_names[0]}: %{{x:.2f}}<br>"
                        f"{feature_names[1]}: %{{y:.2f}}<extra></extra>"
                    ),
                )
            )

    # 3. Held-out Test Data Points
    for c_idx, c_name in enumerate(class_names):
        mask_te = y_test == c_idx
        if np.any(mask_te):
            fig.add_trace(
                go.Scatter(
                    x=X_test_2d[mask_te, 0],
                    y=X_test_2d[mask_te, 1],
                    mode="markers",
                    name=f"{c_name} (Test)",
                    marker={
                        "size": 11,
                        "color": CLASS_COLORS[c_idx % len(CLASS_COLORS)],
                        "symbol": "diamond",
                        "line": {"color": DARK_NAVY, "width": 1.5},
                    },
                    hovertemplate=(
                        f"<b>{c_name} (Test)</b><br>"
                        f"{feature_names[0]}: %{{x:.2f}}<br>"
                        f"{feature_names[1]}: %{{y:.2f}}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        xaxis={"title": f"<b>{feature_names[0]}</b>"},
        yaxis={"title": f"<b>{feature_names[1]}</b>"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return _apply_theme(fig, f"2D KNN Decision Boundary: {feature_names[0]} vs. {feature_names[1]}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    Visualization 3: Interactive Confusion Matrix Heatmap.

    Displays raw counts and row-normalized recall percentages.

    Args:
        cm: Confusion matrix 2D array.
        class_names: List of class labels.

    Returns:
        Plotly Figure.
    """
    # Calculate row-normalized percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = (cm / row_sums) * 100.0

    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if count > (cm.max() * 0.45) else DARK_NAVY
            annotations.append({
                "x": class_names[j],
                "y": class_names[i],
                "text": f"<b>{count}</b><br><span style='font-size:10px;'>({pct:.1f}%)</span>",
                "showarrow": False,
                "font": {"color": color, "size": 13, "family": "Inter, sans-serif"},
            })

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale=[[0, "#F0FDFA"], [0.5, "#5EEAD4"], [1, PRIMARY_TEAL]],
            showscale=False,
            hovertemplate=(
                "True Class: <b>%{y}</b><br>"
                "Predicted Class: <b>%{x}</b><br>"
                "Count: <b>%{z}</b><extra></extra>"
            ),
        )
    )

    fig.update_layout(
        annotations=annotations,
        xaxis={"title": "<b>Predicted Class</b>", "side": "bottom"},
        yaxis={"title": "<b>Actual True Class</b>", "autorange": "reversed"},
    )

    return _apply_theme(fig, "Held-Out Test Set Confusion Matrix")


def plot_before_after(
    default_metrics: Dict[str, float],
    tuned_metrics: Dict[str, float],
    baseline_label: str = "Before (Baseline)",
    tuned_label: str = "After (Tuned)",
) -> go.Figure:
    """
    Visualization 4: Before vs. After Optimization.

    Bar chart comparing Baseline KNN vs. Tuned KNN on the identical held-out test set.

    Args:
        default_metrics: Dict containing accuracy, macro_precision, macro_recall, macro_f1.
        tuned_metrics: Dict containing tuned model metrics.
        baseline_label: Legend label for baseline model.
        tuned_label: Legend label for tuned model.

    Returns:
        Plotly Figure.
    """
    metrics_list = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]

    vals_base = [default_metrics.get(k, 0.0) * 100 for k in keys]
    vals_tuned = [tuned_metrics.get(k, 0.0) * 100 for k in keys]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name=baseline_label,
            x=metrics_list,
            y=vals_base,
            marker_color="#94A3B8",
            text=[f"{v:.1f}%" for v in vals_base],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name=tuned_label,
            x=metrics_list,
            y=vals_tuned,
            marker_color=PRIMARY_TEAL,
            text=[f"{v:.1f}%" for v in vals_tuned],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis={"title": "<b>Score (%)</b>", "range": [0, 105], "ticksuffix": "%"},
        xaxis={"title": "<b>Performance Metric</b>"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return _apply_theme(fig, "Before vs. After Optimization (Held-Out Test Set)")


def plot_dataset_comparison(df_comp: pd.DataFrame) -> go.Figure:
    """
    Visualization 5: Multi-Dataset Benchmark Comparison.

    Displays Test Accuracy and Best K side-by-side across Iris, Wine, Breast Cancer, Digits.

    Args:
        df_comp: DataFrame with ['Dataset', 'Best K', 'Test Accuracy', 'CV Accuracy', 'Macro F1'].

    Returns:
        Plotly Figure with dual subplots.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Held-Out Test Accuracy (%)</b>",
            "<b>Optimal Neighbor Count (Best K)</b>",
        ),
        horizontal_spacing=0.14,
    )

    # 1. Test Accuracy Bar
    fig.add_trace(
        go.Bar(
            x=df_comp["Dataset"],
            y=df_comp["Test Accuracy"] * 100,
            marker_color=PRIMARY_TEAL,
            text=[f"{acc * 100:.1f}%" for acc in df_comp["Test Accuracy"]],
            textposition="auto",
            name="Test Accuracy",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Optimal K Bar
    fig.add_trace(
        go.Bar(
            x=df_comp["Dataset"],
            y=df_comp["Best K"],
            marker_color=ACCENT_ORANGE,
            text=[f"K={k}" for k in df_comp["Best K"]],
            textposition="auto",
            name="Best K",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="<b>Accuracy (%)</b>", range=[0, 105], ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title_text="<b>Best K Value</b>", dtick=2, row=1, col=2)

    return _apply_theme(fig, "Multi-Dataset KNN Benchmark Comparison")


def plot_prediction_probabilities(
    probabilities: np.ndarray,
    class_names: List[str],
    predicted_idx: int,
) -> go.Figure:
    """
    Plot horizontal probability bars for a sample prediction.

    Args:
        probabilities: 1D array of class probabilities summing to 1.0.
        class_names: Names of classes.
        predicted_idx: Index of highest-confidence class.

    Returns:
        Plotly Figure.
    """
    pcts = probabilities * 100.0
    colors = [
        ACCENT_ORANGE if idx == predicted_idx else PRIMARY_TEAL
        for idx in range(len(class_names))
    ]

    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=class_names,
            orientation="h",
            marker={"color": colors},
            text=[f"{p:.1f}%" for p in pcts],
            textposition="auto",
        )
    )

    fig.update_layout(
        xaxis={"title": "<b>Class Probability (%)</b>", "range": [0, 105], "ticksuffix": "%"},
        yaxis={"title": "<b>Target Class</b>", "autorange": "reversed"},
    )

    return _apply_theme(fig, "Prediction Confidence Distribution")


def plot_training_vs_cv(df_k: pd.DataFrame) -> go.Figure:
    """
    Dual curve: Training Accuracy vs. 5-Fold Cross-Validation Accuracy across K.
    Demonstrates overfitting (K=1) vs underfitting (large K).

    Args:
        df_k: DataFrame containing ['k', 'cv_mean', 'train_acc'].

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_k["k"],
            y=df_k["train_acc"] * 100,
            mode="lines+markers",
            name="Training Accuracy",
            line={"color": "#EF4444", "width": 2.5, "dash": "dot"},
            marker={"size": 7, "color": "#EF4444", "symbol": "square"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_k["k"],
            y=df_k["cv_mean"] * 100,
            mode="lines+markers",
            name="Cross-Validation Accuracy",
            line={"color": PRIMARY_TEAL, "width": 3},
            marker={"size": 8, "color": PRIMARY_TEAL, "symbol": "circle"},
        )
    )

    fig.update_layout(
        xaxis={"title": "<b>Number of Neighbors (K)</b>", "dtick": 1},
        yaxis={"title": "<b>Accuracy (%)</b>", "ticksuffix": "%"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return _apply_theme(fig, "Overfitting vs. Underfitting: Training vs. Validation Accuracy")


def plot_class_distribution(y: pd.Series, class_names: List[str]) -> go.Figure:
    """
    Bar chart showing sample counts per class in the dataset.

    Args:
        y: Series of class indices.
        class_names: List of class strings.

    Returns:
        Plotly Figure.
    """
    counts = y.value_counts().sort_index()
    labels = [class_names[idx] if idx < len(class_names) else str(idx) for idx in counts.index]
    values = counts.values

    colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(labels))]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v} samples" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        xaxis={"title": "<b>Class Name</b>"},
        yaxis={"title": "<b>Number of Samples</b>"},
    )

    return _apply_theme(fig, "Dataset Class Distribution")


def plot_digit_preview(image_8x8: np.ndarray, title: str = "Digit Bitmap") -> go.Figure:
    """
    Heatmap rendering of an 8x8 handwritten digit bitmap.

    Args:
        image_8x8: 2D array of shape (8, 8).
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=image_8x8,
            colorscale="Greys",
            reversescale=True,
            showscale=False,
            hoverinfo="z",
        )
    )

    fig.update_layout(
        xaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"showticklabels": False, "showgrid": False, "zeroline": False, "autorange": "reversed"},
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        width=220,
        height=220,
    )

    return _apply_theme(fig, title)
