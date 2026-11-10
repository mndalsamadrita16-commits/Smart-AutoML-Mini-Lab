"""
SMART ML PARAMETER TUNER — AutoML Mini Lab
Interactive K-Nearest Neighbors (KNN) classification laboratory.

A complete, production-grade Streamlit application for AutoML parameter tuning,
stratified cross-validation, decision boundary visualization, and model comparison.

All calculations, metrics, charts, and predictions are dynamically computed from real data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path to guarantee clean imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DatasetBundle, get_dataset_summary, load_dataset
from src.evaluation import (
    compare_baseline_vs_tuned,
    evaluate_pipeline,
    run_baseline_model,
    run_multi_dataset_benchmark,
    run_standardization_experiment,
)
from src.model import (
    compute_accuracy_vs_k,
    get_k_nearest_neighbors,
    run_grid_search,
    train_2d_model,
)
from src.preprocessing import build_pipeline, split_data
from src.utils import export_results_dataframe, serialize_params
from src.visualization import (
    plot_accuracy_vs_k,
    plot_before_after,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_dataset_comparison,
    plot_decision_boundary,
    plot_digit_preview,
    plot_prediction_probabilities,
    plot_training_vs_cv,
)

# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLES
# ==============================================================================

st.set_page_config(
    page_title="SMART ML PARAMETER TUNER — AutoML Mini Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
}

.stApp {
    background-color: #F8FAFC;
}

/* Header Container */
.header-wrapper {
    background: linear-gradient(135deg, #1B3A57 0%, #1B8A93 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 15px rgba(27, 58, 87, 0.12);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-title-box h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
}

.header-title-box p {
    margin: 6px 0 0 0;
    font-size: 1.05rem;
    color: #E2E8F0;
    font-weight: 400;
}

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 18px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.status-badge-trained {
    background-color: #ECFDF5;
    color: #065F46;
    border: 1.5px solid #10B981;
}

.status-badge-ready {
    background-color: #EFF6FF;
    color: #1E40AF;
    border: 1.5px solid #3B82F6;
}

.status-badge-notrun {
    background-color: #FFFBEB;
    color: #92400E;
    border: 1.5px solid #F59E0B;
}

/* Information Container */
.info-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #1B8A93;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.info-card-orange {
    border-left-color: #E0A040;
}

.info-card-navy {
    border-left-color: #1B3A57;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1B3A57;
    margin: 18px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Streamlit Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 2px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    font-weight: 600;
    font-size: 0.92rem;
    border-radius: 8px 8px 0 0;
    color: #64748B;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    color: #1B8A93 !important;
    border-bottom: 3px solid #1B8A93 !important;
    background-color: #F0FDFA;
}

/* Custom Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"] {
    background-color: #1B8A93;
    border-color: #1B8A93;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background-color: #146C73;
    border-color: #146C73;
}

/* Footer styling */
.app-footer {
    text-align: center;
    padding: 24px 0 12px 0;
    margin-top: 48px;
    border-top: 1px solid #E2E8F0;
    color: #64748B;
    font-size: 0.88rem;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & ML FUNCTIONS
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_cached_dataset(dataset_name: str) -> DatasetBundle:
    """Load and cache authentic classification dataset from sklearn."""
    return load_dataset(dataset_name)


@st.cache_data(show_spinner=False)
def get_cached_accuracy_vs_k(
    dataset_name: str,
    test_size: float,
    seed: int,
    metric: str,
    weights: str,
    standardize: bool,
    p: int = 2,
    max_k: int = 15,
) -> pd.DataFrame:
    """Compute and cache Accuracy vs K curve on training partition."""
    bundle = get_cached_dataset(dataset_name)
    X_tr, _, y_tr, _ = split_data(bundle.X, bundle.y, test_size=test_size, random_state=seed)
    return compute_accuracy_vs_k(
        X_train=X_tr,
        y_train=y_tr,
        metric=metric,
        weights=weights,
        standardize=standardize,
        p=p,
        max_k=max_k,
        cv_splits=5,
        random_state=seed,
    )


@st.cache_data(show_spinner=False)
def get_cached_grid_search(
    dataset_name: str,
    test_size: float,
    seed: int,
    max_k: int = 15,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Run and cache GridSearchCV strictly on training data.
    Returns: (cleaned_best_params, best_cv_score, test_evaluation_dict)
    """
    bundle = get_cached_dataset(dataset_name)
    X_tr, X_te, y_tr, y_te = split_data(bundle.X, bundle.y, test_size=test_size, random_state=seed)

    grid, best_params = run_grid_search(
        X_tr,
        y_tr,
        max_k=max_k,
        metrics=("euclidean", "manhattan", "minkowski"),
        weights=("uniform", "distance"),
        tune_scaling=True,
        cv_splits=5,
        random_state=seed,
    )
    best_estimator = grid.best_estimator_

    test_eval = evaluate_pipeline(
        best_estimator,
        X_tr,
        y_tr,
        X_te,
        y_te,
        bundle.class_names,
        cv_folds=5,
        random_state=seed,
    )

    return best_params, float(grid.best_score_), test_eval


@st.cache_data(show_spinner=False)
def get_cached_multi_dataset_benchmark(test_size: float, seed: int) -> List[Dict[str, Any]]:
    """Run and cache independent ML pipelines across all 4 datasets."""
    return run_multi_dataset_benchmark(test_size=test_size, random_state=seed)


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_header(status_state: str) -> None:
    """Render top application header with dynamic status badge."""
    if status_state == "MODEL_TRAINED":
        badge_html = '<div class="status-badge status-badge-trained">● MODEL TRAINED</div>'
    elif status_state == "OPTIMIZING":
        badge_html = '<div class="status-badge status-badge-ready">● OPTIMIZING...</div>'
    else:
        badge_html = '<div class="status-badge status-badge-ready">● MODEL READY</div>'

    st.markdown(
        f"""
        <div class="header-wrapper">
            <div class="header-title-box">
                <h1>SMART ML PARAMETER TUNER</h1>
                <p>AutoML Mini Lab &mdash; Interactive KNN Classification & Hyperparameter Optimization</p>
            </div>
            <div>
                {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bundle: DatasetBundle) -> Dict[str, Any]:
    """Render sidebar controls and return user configuration."""
    st.sidebar.markdown("## ⚙️ Model Controls")

    # 1. Dataset Selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        options=["Iris", "Wine", "Breast Cancer", "Digits"],
        index=0,
        help="Select a benchmark dataset from scikit-learn. Iris is default.",
    )

    # 2. Manual Hyperparameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Hyperparameter Settings")

    manual_k = st.sidebar.slider(
        "K Value (Number of Neighbors)",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="Number of nearest neighbors to consider for majority voting (1 to 15).",
    )

    metric_display = st.sidebar.selectbox(
        "Distance Metric",
        options=["Euclidean", "Manhattan", "Minkowski"],
        index=0,
        help="Distance metric used to calculate proximity in feature space.",
    )

    metric_map = {
        "Euclidean": "euclidean",
        "Manhattan": "manhattan",
        "Minkowski": "minkowski",
    }
    distance_metric = metric_map[metric_display]

    minkowski_p = 2
    if distance_metric == "minkowski":
        minkowski_p = st.sidebar.slider(
            "Minkowski p-value",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Power parameter p for Minkowski distance (p=1 is Manhattan, p=2 is Euclidean).",
        )

    weights = st.sidebar.selectbox(
        "Weight Function",
        options=["uniform", "distance"],
        index=0,
        help="uniform: all neighbors vote equally. distance: weight by inverse of distance.",
    )

    standardize = st.sidebar.checkbox(
        "Standardize Features (StandardScaler)",
        value=True,
        help="Scale features to zero mean and unit variance inside the Pipeline.",
    )

    # 3. Train/Test Split
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Validation Split")

    test_size_pct = st.sidebar.slider(
        "Test Split (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data held out untouched for final evaluation (default: 20%).",
    )
    test_size = test_size_pct / 100.0

    random_seed = st.sidebar.number_input(
        "Random State Seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Random seed for reproducible stratified train/test split and CV fold shuffling.",
    )

    # 4. Decision Boundary Features
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ 2D Decision Boundary Features")

    feature_options = bundle.feature_names
    feat_x = st.sidebar.selectbox(
        "Feature X",
        options=feature_options,
        index=0,
        help="Select X-axis feature for 2D decision boundary visualization.",
    )

    default_y_idx = 1 if len(feature_options) > 1 else 0
    feat_y = st.sidebar.selectbox(
        "Feature Y",
        options=feature_options,
        index=default_y_idx,
        help="Select Y-axis feature for 2D decision boundary visualization.",
    )

    # 5. Run Optimization Trigger
    st.sidebar.markdown("---")
    run_opt = st.sidebar.button(
        "🚀 Run Optimization",
        type="primary",
        width="stretch",
        help="Execute 5-fold Stratified Cross-Validation on training data across K=1..15 and distance metrics.",
    )

    # Sidebar Model Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Model Status")
    is_opt = st.session_state.get("optimization_completed", False)
    st.sidebar.info(
        f"**Training Status:** {'Optimized & Trained' if is_opt else 'Manual Model Active'}\n\n"
        f"**Active Model:** KNN (K={st.session_state.get('active_k', manual_k)})\n\n"
        f"**Metric:** {distance_metric.capitalize()}\n\n"
        f"**Best Parameters:** {st.session_state.get('best_param_str', 'Pending Optimization')}"
    )

    # Sidebar About
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.caption(
        "**SMART ML PARAMETER TUNER**\n\n"
        "AutoML Mini Lab &middot; Visual K-Nearest Neighbors Classification Laboratory.\n\n"
        "Evaluates real models with zero data leakage using 5-Fold Stratified Cross-Validation."
    )

    return {
        "dataset_name": dataset_name,
        "manual_k": manual_k,
        "distance_metric": distance_metric,
        "minkowski_p": minkowski_p,
        "weights": weights,
        "standardize": standardize,
        "test_size": test_size,
        "random_seed": int(random_seed),
        "feat_x": feat_x,
        "feat_y": feat_y,
        "run_opt": run_opt,
    }


def render_performance_cards(
    bundle: DatasetBundle,
    sidebar_cfg: Dict[str, Any],
    manual_eval: Dict[str, Any],
    optimized_params: Optional[Dict[str, Any]] = None,
    optimized_eval: Optional[Dict[str, Any]] = None,
    optimized_cv: Optional[float] = None,
) -> None:
    """
    Render top performance cards:
    Dataset | Algorithm | Selected K | Best K | CV Accuracy | Test Accuracy | Distance Metric | Number of Features
    """
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

    col1.metric("Dataset", bundle.name)
    col2.metric("Algorithm", "KNN Classifier")
    col3.metric("Selected K", str(sidebar_cfg["manual_k"]))

    if optimized_params is not None and optimized_eval is not None and optimized_cv is not None:
        best_k_str = str(optimized_params.get("n_neighbors", "—"))
        cv_acc_str = f"{optimized_cv * 100:.2f}%"
        test_acc_str = f"{optimized_eval['test_metrics']['accuracy'] * 100:.2f}%"
        active_metric_str = optimized_params.get("metric", sidebar_cfg["distance_metric"]).capitalize()
    else:
        best_k_str = "Pending"
        cv_acc_str = f"{manual_eval['cv_mean'] * 100:.2f}%"
        test_acc_str = f"{manual_eval['test_metrics']['accuracy'] * 100:.2f}%"
        active_metric_str = sidebar_cfg["distance_metric"].capitalize()

    col4.metric("Best K", best_k_str)
    col5.metric("CV Accuracy", cv_acc_str)
    col6.metric("Test Accuracy", test_acc_str)
    col7.metric("Distance Metric", active_metric_str)
    col8.metric("Features", f"{bundle.X.shape[1]}")


# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================

def render_dashboard_tab(
    bundle: DatasetBundle,
    sidebar_cfg: Dict[str, Any],
    manual_eval: Dict[str, Any],
    optimized_params: Optional[Dict[str, Any]],
    optimized_eval: Optional[Dict[str, Any]],
    optimized_cv: Optional[float],
) -> None:
    """Render executive overview dashboard with model specifications and data preview."""
    st.markdown('<div class="section-title">📊 Executive Dashboard & Lab Status</div>', unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([1.5, 1.3, 1.2])

    with col_left:
        st.markdown(
            f"""
            <div class="info-card">
                <h4 style="margin-top:0; color:#1B3A57;">Active Dataset: {bundle.name}</h4>
                <p style="font-size:0.9rem; color:#475569;">{bundle.description}</p>
                <div style="font-size:0.88rem; line-height:1.7;">
                    &bull; <b>Total Samples:</b> {bundle.X.shape[0]:,}<br>
                    &bull; <b>Dimensional Features:</b> {bundle.X.shape[1]}<br>
                    &bull; <b>Target Classes ({len(bundle.class_names)}):</b> {', '.join(bundle.class_names)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mid:
        best_k_display = str(optimized_params["n_neighbors"]) if optimized_params else "Pending Optimization"
        best_metric_display = optimized_params["metric"].capitalize() if optimized_params else "Pending Optimization"
        scaler_display = "StandardScaler" if sidebar_cfg["standardize"] else "None (passthrough)"

        st.markdown(
            f"""
            <div class="info-card info-card-navy">
                <h4 style="margin-top:0; color:#1B3A57;">Experimental Specifications</h4>
                <div style="font-size:0.86rem; line-height:1.7;">
                    &bull; <b>Algorithm:</b> K-Nearest Neighbors (KNN)<br>
                    &bull; <b>Preprocessing:</b> {scaler_display}<br>
                    &bull; <b>Validation Protocol:</b> 5-Fold Stratified Cross-Validation<br>
                    &bull; <b>Parameter Search Space:</b> K = 1 through 15<br>
                    &bull; <b>Test Partition:</b> {int(sidebar_cfg['test_size'] * 100)}% Held-Out (Untouched)<br>
                    &bull; <b>Optimal K Found:</b> <code>{best_k_display}</code> ({best_metric_display})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="info-card info-card-orange">
                <h4 style="margin-top:0; color:#1B3A57;">💡 KNN Principles</h4>
                <div style="font-size:0.85rem; color:#334155; line-height:1.6;">
                    &bull; <b>Distance Based:</b> Closeness in n-dimensional space determines class.<br>
                    &bull; <b>Small K (e.g. 1):</b> High variance, sensitive to noise, risks overfitting.<br>
                    &bull; <b>Large K:</b> High bias, smoother boundaries, risks underfitting.<br>
                    &bull; <b>Scaling Essential:</b> Prevents large features from dominating distances.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Dataset preview
    st.markdown('<div class="section-title">🔍 Dataset Preview</div>', unsafe_allow_html=True)
    df_preview = bundle.X.copy()
    df_preview["Target_Class"] = [bundle.class_names[idx] for idx in bundle.y]
    st.dataframe(df_preview.head(10), width="stretch")


# ==============================================================================
# TAB 2: OPTIMIZATION (ACCURACY VS K)
# ==============================================================================

def render_optimization_tab(
    bundle: DatasetBundle,
    sidebar_cfg: Dict[str, Any],
    optimized_params: Optional[Dict[str, Any]],
) -> None:
    """Render mandatory Accuracy vs K validation curve and full evaluation table."""
    st.markdown('<div class="section-title">📈 Cross-Validation Accuracy vs. Number of Neighbors (K)</div>', unsafe_allow_html=True)
    st.caption("5-Fold Stratified Cross-Validation evaluated strictly on the training partition across K = 1 through 15.")

    with st.spinner("Computing 5-Fold Stratified Cross-Validation across K = 1..15..."):
        df_k = get_cached_accuracy_vs_k(
            dataset_name=bundle.name,
            test_size=sidebar_cfg["test_size"],
            seed=sidebar_cfg["random_seed"],
            metric=sidebar_cfg["distance_metric"],
            weights=sidebar_cfg["weights"],
            standardize=sidebar_cfg["standardize"],
            p=sidebar_cfg["minkowski_p"],
            max_k=15,
        )

    best_k_val = optimized_params.get("n_neighbors") if optimized_params else int(df_k.loc[df_k["cv_mean"].idxmax()]["k"])

    c_chart, c_train = st.columns([1.2, 1.0])

    with c_chart:
        fig_acc_k = plot_accuracy_vs_k(df_k, best_k=best_k_val)
        st.plotly_chart(fig_acc_k, width="stretch", key="fig_accuracy_vs_k")

    with c_train:
        fig_train_cv = plot_training_vs_cv(df_k)
        st.plotly_chart(fig_train_cv, width="stretch", key="fig_train_vs_cv")

    # Dynamic Insights
    best_row = df_k.loc[df_k["cv_mean"].idxmax()]
    train_k1 = float(df_k.loc[df_k["k"] == 1]["train_acc"].iloc[0])
    cv_k1 = float(df_k.loc[df_k["k"] == 1]["cv_mean"].iloc[0])

    st.markdown(
        f"""
        <div class="info-card">
            <b>Computed Cross-Validation Insights for {bundle.name}:</b><br>
            &bull; <b>Optimal K:</b> Cross-validation accuracy peaks at <b>K = {int(best_row['k'])}</b> with <b>{best_row['cv_mean'] * 100:.2f}% (&plusmn;{best_row['cv_std'] * 100:.2f}%)</b>.<br>
            &bull; <b>Variance at K = 1:</b> Training accuracy is <b>{train_k1 * 100:.1f}%</b> while CV accuracy is <b>{cv_k1 * 100:.1f}%</b>. This confirms the classic memorization phenomenon of 1-NN.<br>
            &bull; <b>Current Selected K:</b> K = {sidebar_cfg['manual_k']} yields a CV score of <b>{float(df_k.loc[df_k['k'] == sidebar_cfg['manual_k']]['cv_mean'].iloc[0]) * 100:.2f}%</b> (Rank #{int(df_k.loc[df_k['k'] == sidebar_cfg['manual_k']]['rank'].iloc[0])}).
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 View Complete K = 1..15 Evaluation Table"):
        display_df = df_k.copy()
        display_df["cv_mean"] = display_df["cv_mean"].apply(lambda v: f"{v * 100:.2f}%")
        display_df["cv_std"] = display_df["cv_std"].apply(lambda v: f"±{v * 100:.2f}%")
        display_df["train_acc"] = display_df["train_acc"].apply(lambda v: f"{v * 100:.2f}%")
        display_df.columns = ["K", "Mean CV Accuracy", "CV Std Dev", "Training Accuracy", "Rank"]
        st.dataframe(display_df, width="stretch", hide_index=True)


# ==============================================================================
# TAB 3: DECISION BOUNDARY
# ==============================================================================

def render_decision_boundary_tab(
    bundle: DatasetBundle,
    sidebar_cfg: Dict[str, Any],
) -> None:
    """Render mandatory 2D KNN decision boundary with mesh contours and scatter points."""
    st.markdown('<div class="section-title">🗺️ Interactive 2D KNN Decision Boundary</div>', unsafe_allow_html=True)

    feat_x = sidebar_cfg["feat_x"]
    feat_y = sidebar_cfg["feat_y"]

    if feat_x == feat_y:
        st.warning("⚠️ Please select two distinct features in the sidebar to visualize the 2D decision boundary.")
        return

    # Extract 2D projection
    X_2d = bundle.X[[feat_x, feat_y]].copy()
    X_tr_2d, X_te_2d, y_tr_2d, y_te_2d = split_data(
        X_2d,
        bundle.y,
        test_size=sidebar_cfg["test_size"],
        random_state=sidebar_cfg["random_seed"],
    )

    # Train 2D KNN model strictly for visualization
    vis_pipe = train_2d_model(
        X_train_2d=X_tr_2d.values,
        y_train=y_tr_2d.values,
        feature_names=[feat_x, feat_y],
        n_neighbors=sidebar_cfg["manual_k"],
        metric=sidebar_cfg["distance_metric"],
        weights=sidebar_cfg["weights"],
        standardize=sidebar_cfg["standardize"],
        p=sidebar_cfg["minkowski_p"],
    )

    fig_boundary = plot_decision_boundary(
        model_2d=vis_pipe,
        X_train_2d=X_tr_2d.values,
        y_train=y_tr_2d.values,
        X_test_2d=X_te_2d.values,
        y_test=y_te_2d.values,
        feature_names=[feat_x, feat_y],
        class_names=bundle.class_names,
    )

    st.plotly_chart(fig_boundary, width="stretch", key="fig_decision_boundary")

    st.info(
        f"📌 **Interpretation:** Colored classification regions represent decision zones produced by KNN (K={sidebar_cfg['manual_k']}, metric={sidebar_cfg['distance_metric']}). "
        f"**Circle markers** represent training samples ({len(X_tr_2d)}); **Diamond markers** represent held-out test samples ({len(X_te_2d)}). "
        f"This 2D model is trained specifically on '{feat_x}' and '{feat_y}' for visualization purposes."
    )


# ==============================================================================
# TAB 4: CONFUSION MATRIX & METRICS
# ==============================================================================

def render_confusion_matrix_tab(
    bundle: DatasetBundle,
    evaluation: Dict[str, Any],
    model_label: str = "Trained Model",
) -> None:
    """Render mandatory held-out test confusion matrix and comprehensive classification metrics."""
    st.markdown(f'<div class="section-title">🎯 Confusion Matrix & Test Metrics ({model_label})</div>', unsafe_allow_html=True)
    st.caption("Calculated strictly on the held-out test set using actual predictions from the trained model.")

    test_metrics = evaluation["test_metrics"]
    cm = evaluation["confusion_matrix"]
    report_df = evaluation["report_df"]

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{test_metrics['accuracy'] * 100:.2f}%")
    m2.metric("Macro Precision", f"{test_metrics['macro_precision'] * 100:.2f}%")
    m3.metric("Macro Recall", f"{test_metrics['macro_recall'] * 100:.2f}%")
    m4.metric("Macro F1-Score", f"{test_metrics['macro_f1'] * 100:.2f}%")

    col_cm, col_rep = st.columns([1.1, 1.2])

    with col_cm:
        fig_cm = plot_confusion_matrix(cm, bundle.class_names)
        st.plotly_chart(fig_cm, width="stretch", key="fig_confusion_matrix")

    with col_rep:
        st.markdown("##### Detailed Classification Report")
        st.dataframe(
            report_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }),
            width="stretch",
        )

        csv_bytes = report_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Classification Report CSV",
            data=csv_bytes,
            file_name=f"{bundle.name.lower().replace(' ', '_')}_classification_report.csv",
            mime="text/csv",
            key="btn_download_cm_csv",
        )


# ==============================================================================
# TAB 5: MODEL COMPARISON
# ==============================================================================

def render_comparison_tab(
    bundle: DatasetBundle,
    sidebar_cfg: Dict[str, Any],
    default_eval: Dict[str, Any],
    optimized_eval: Optional[Dict[str, Any]],
    optimized_params: Optional[Dict[str, Any]],
    manual_eval: Dict[str, Any],
) -> None:
    """Render Before vs. After Optimization and Multi-Dataset Benchmark comparisons."""
    st.markdown('<div class="section-title">⚖️ Part 1: Before vs. After Optimization</div>', unsafe_allow_html=True)
    st.caption("Compares Default Baseline KNN (K=5, Euclidean, unscaled) against Tuned KNN evaluated on the identical held-out test partition.")

    if optimized_eval is None or optimized_params is None:
        st.info("ℹ️ Click **'Run Optimization'** in the sidebar to compute GridSearchCV parameters and activate this comparison.")
    else:
        comp_res = compare_baseline_vs_tuned(default_eval, optimized_eval)

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Test Accuracy (K=5)", f"{comp_res['baseline_accuracy'] * 100:.2f}%")
        c2.metric(
            f"Tuned Test Accuracy (K={optimized_params['n_neighbors']})",
            f"{comp_res['tuned_accuracy'] * 100:.2f}%",
            delta=f"{comp_res['accuracy_diff_pct']:+.2f}%",
        )
        c3.metric(
            "Macro F1 Improvement",
            f"{comp_res['tuned_f1'] * 100:.2f}%",
            delta=f"{(comp_res['tuned_f1'] - comp_res['baseline_f1']) * 100:+.2f}%",
        )

        col_plot, col_table = st.columns([1.3, 1.2])

        with col_plot:
            fig_bar = plot_before_after(
                default_eval["test_metrics"],
                optimized_eval["test_metrics"],
                baseline_label="Baseline (K=5, Euclidean)",
                tuned_label=f"Tuned (K={optimized_params['n_neighbors']}, {optimized_params['metric'].capitalize()})",
            )
            st.plotly_chart(fig_bar, width="stretch", key="fig_before_after")

        with col_table:
            st.markdown("##### Parameter & Performance Audit")
            audit_df = pd.DataFrame({
                "Specification": [
                    "K (Neighbors)",
                    "Distance Metric",
                    "Neighbor Weights",
                    "Feature Scaling",
                    "CV Accuracy",
                    "Held-Out Test Accuracy",
                    "Held-Out Macro F1",
                ],
                "Baseline Model": [
                    "5",
                    "euclidean",
                    "uniform",
                    "None (passthrough)",
                    f"{default_eval['cv_mean'] * 100:.2f}%",
                    f"{default_eval['test_metrics']['accuracy'] * 100:.2f}%",
                    f"{default_eval['test_metrics']['macro_f1'] * 100:.2f}%",
                ],
                "Tuned Model": [
                    str(optimized_params["n_neighbors"]),
                    optimized_params["metric"],
                    optimized_params["weights"],
                    optimized_params["scaler"],
                    f"{optimized_params['cv_score'] * 100:.2f}%",
                    f"{optimized_eval['test_metrics']['accuracy'] * 100:.2f}%",
                    f"{optimized_eval['test_metrics']['macro_f1'] * 100:.2f}%",
                ],
            })
            st.dataframe(audit_df, width="stretch", hide_index=True)

    # Part 2: Multi-dataset comparison
    st.markdown("---")
    st.markdown('<div class="section-title">🌐 Part 2: Multi-Dataset Benchmark Comparison</div>', unsafe_allow_html=True)
    st.caption("Executes independent pipelines and Stratified GridSearchCV optimizations across Iris, Wine, Breast Cancer, and Digits.")

    btn_multi = st.button("🔄 Execute Multi-Dataset Benchmark", key="btn_run_multi_benchmark")

    if btn_multi or "multi_bench_results" in st.session_state:
        if btn_multi or "multi_bench_results" not in st.session_state:
            with st.spinner("Training independent ML pipelines and optimizing across all 4 datasets..."):
                bench_data = get_cached_multi_dataset_benchmark(
                    test_size=sidebar_cfg["test_size"],
                    seed=sidebar_cfg["random_seed"],
                )
                st.session_state["multi_bench_results"] = bench_data
        else:
            bench_data = st.session_state["multi_bench_results"]

        df_bench = pd.DataFrame(bench_data)

        fig_bench = plot_dataset_comparison(df_bench)
        st.plotly_chart(fig_bench, width="stretch", key="fig_dataset_comparison")

        st.markdown("##### 📋 Multi-Dataset Results Summary")
        disp_bench = df_bench.copy()
        disp_bench["CV Accuracy"] = disp_bench["CV Accuracy"].apply(lambda v: f"{v * 100:.2f}%")
        disp_bench["Test Accuracy"] = disp_bench["Test Accuracy"].apply(lambda v: f"{v * 100:.2f}%")
        disp_bench["Macro F1"] = disp_bench["Macro F1"].apply(lambda v: f"{v * 100:.2f}%")
        st.dataframe(disp_bench, width="stretch", hide_index=True)


# ==============================================================================
# TAB 6: PREDICTION
# ==============================================================================

def render_prediction_tab(
    bundle: DatasetBundle,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int,
) -> None:
    """Render interactive prediction interface with input fields, prediction probabilities, and nearest neighbors."""
    st.markdown('<div class="section-title">🔮 Predict a New Sample</div>', unsafe_allow_html=True)

    actual_class_str: Optional[str] = None
    raw_digit_img: Optional[np.ndarray] = None

    if bundle.name == "Iris":
        st.markdown("Adjust the 4 morphological feature sliders below to predict flower species:")
        p1, p2, p3, p4 = st.columns(4)
        input_dict: Dict[str, float] = {}

        cols = [p1, p2, p3, p4]
        for col, feat in zip(cols, bundle.feature_names):
            min_v = float(bundle.X[feat].min())
            max_v = float(bundle.X[feat].max())
            mean_v = float(bundle.X[feat].mean())
            with col:
                input_dict[feat] = st.slider(
                    feat,
                    min_value=round(min_v * 0.8, 1),
                    max_value=round(max_v * 1.2, 1),
                    value=round(mean_v, 1),
                    step=0.1,
                )
        sample_vec = pd.DataFrame([input_dict])

    elif bundle.name == "Digits":
        st.markdown("Select a handwritten digit sample from the held-out test partition to classify:")
        col_btn, col_num = st.columns([1, 2])

        with col_btn:
            if st.button("🎲 Random Test Sample", key="btn_random_digit"):
                st.session_state["digit_idx"] = int(np.random.choice(len(X_test)))

        if "digit_idx" not in st.session_state:
            st.session_state["digit_idx"] = 0

        with col_num:
            digit_pos = st.number_input(
                f"Test Sample Index (0 to {len(X_test) - 1})",
                min_value=0,
                max_value=len(X_test) - 1,
                value=st.session_state["digit_idx"],
                step=1,
            )

        chosen_row = X_test.iloc[digit_pos]
        sample_vec = chosen_row.to_frame().T
        true_label_idx = y_test.iloc[digit_pos]
        actual_class_str = bundle.class_names[true_label_idx]

        if bundle.raw_images is not None:
            original_idx = int(X_test.index[digit_pos])
            raw_digit_img = bundle.raw_images[original_idx]

    else:
        st.markdown(f"Generate predictions for the high-dimensional **{bundle.name}** dataset:")
        col_btn, col_num = st.columns([1, 2])

        with col_btn:
            if st.button("🎲 Random Test Sample", key="btn_random_sample"):
                st.session_state["sample_idx"] = int(np.random.choice(len(X_test)))

        if "sample_idx" not in st.session_state:
            st.session_state["sample_idx"] = 0

        with col_num:
            sample_pos = st.number_input(
                f"Test Sample Index (0 to {len(X_test) - 1})",
                min_value=0,
                max_value=len(X_test) - 1,
                value=st.session_state["sample_idx"],
                step=1,
            )

        chosen_row = X_test.iloc[sample_pos]
        sample_vec = chosen_row.to_frame().T
        true_label_idx = y_test.iloc[sample_pos]
        actual_class_str = bundle.class_names[true_label_idx]

    # Predict Class Button
    predict_btn = st.button("⚡ Predict Class", type="primary", key="btn_predict_class")

    # Always compute prediction if button clicked or sample exists
    pred_idx = int(pipeline.predict(sample_vec)[0])
    pred_class = bundle.class_names[pred_idx]

    try:
        probabilities = pipeline.predict_proba(sample_vec)[0]
    except Exception:
        probabilities = np.zeros(len(bundle.class_names))
        probabilities[pred_idx] = 1.0

    st.markdown("---")
    res_col1, res_col2 = st.columns([1.1, 1.8])

    with res_col1:
        st.markdown(
            f"""
            <div class="info-card info-card-orange">
                <h4 style="margin-top:0; color:#1B3A57;">Predicted Class</h4>
                <div style="font-size:1.6rem; font-weight:700; color:#1B8A93; margin-bottom:8px;">
                    {pred_class}
                </div>
                <div style="font-size:0.92rem; color:#475569;">
                    <b>Confidence:</b> {probabilities[pred_idx] * 100:.1f}%<br>
                    {f'<b>Ground Truth:</b> {actual_class_str}' if actual_class_str else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if raw_digit_img is not None:
            fig_dig = plot_digit_preview(raw_digit_img, title=f"True Digit: {actual_class_str}")
            st.plotly_chart(fig_dig, width="stretch", key="fig_pred_digit")

    with res_col2:
        fig_prob = plot_prediction_probabilities(probabilities, bundle.class_names, pred_idx)
        st.plotly_chart(fig_prob, width="stretch", key="fig_pred_prob")

    # K Nearest Neighbors table
    st.markdown(f"##### 👥 K = {k} Nearest Neighbors in Feature Space")
    neighbors_df = get_k_nearest_neighbors(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        input_sample=sample_vec.values[0],
        n_neighbors=k,
        class_names=bundle.class_names,
    )
    st.dataframe(neighbors_df, width="stretch", hide_index=True)


# ==============================================================================
# TAB 7: DATASET INFORMATION
# ==============================================================================

def render_dataset_tab(bundle: DatasetBundle) -> None:
    """Render comprehensive dataset details, class distribution, and statistics."""
    st.markdown(f'<div class="section-title">📁 Dataset Information: {bundle.name}</div>', unsafe_allow_html=True)

    summary = get_dataset_summary(bundle)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Samples", f"{summary['n_samples']:,}")
    d2.metric("Total Features", f"{summary['n_features']}")
    d3.metric("Number of Classes", f"{summary['n_classes']}")
    d4.metric("Missing Values", f"{summary['missing_values']}")

    col_chart, col_stats = st.columns([1.2, 1.4])

    with col_chart:
        fig_dist = plot_class_distribution(bundle.y, bundle.class_names)
        st.plotly_chart(fig_dist, width="stretch", key="fig_class_distribution")

    with col_stats:
        st.markdown("##### Feature Summary Statistics")
        st.dataframe(bundle.X.describe().transpose()[["mean", "std", "min", "max"]].round(3), width="stretch")

    st.markdown("##### Complete Feature List")
    st.write(", ".join([f"`{col}`" for col in bundle.feature_names]))


# ==============================================================================
# TAB 8: RESEARCH & THEORY
# ==============================================================================

def render_research_tab(
    bundle: DatasetBundle,
    sidebar_cfg: Dict[str, Any],
    default_eval: Dict[str, Any],
    optimized_eval: Optional[Dict[str, Any]],
    optimized_params: Optional[Dict[str, Any]],
) -> None:
    """Render educational KNN theory and the four research questions with live computed evidence."""
    st.markdown('<div class="section-title">🔬 Machine Learning Theory & Empirical Research</div>', unsafe_allow_html=True)

    # Educational Theory Section
    st.markdown("### 📚 How K-Nearest Neighbors (KNN) Works")
    st.markdown(
        """
        K-Nearest Neighbors is a non-parametric, instance-based supervised learning algorithm that makes decisions based on geometric proximity:
        1. **Store Training Data:** Memorizes feature coordinates of all training instances.
        2. **Compute Distances:** For any unlabelled query point, computes distance to all training points using the chosen metric (e.g. Euclidean, Manhattan, Minkowski).
        3. **Select K Neighbors:** Identifies the $K$ points in the training set with the smallest calculated distances.
        4. **Majority Voting:** Conducts majority voting among the $K$ neighbors (or weighted voting inversely proportional to distance).
        5. **Assign Class:** Assigns the winning majority class to the query point.
        """
    )

    # 4 Empirical Research Questions
    st.markdown("---")
    st.markdown("### 🧪 Research Inquiries Answered with Live Evidence")

    # Q1
    st.markdown("#### Q1: How does K affect underfitting and overfitting?")
    df_k = get_cached_accuracy_vs_k(
        dataset_name=bundle.name,
        test_size=sidebar_cfg["test_size"],
        seed=sidebar_cfg["random_seed"],
        metric=sidebar_cfg["distance_metric"],
        weights=sidebar_cfg["weights"],
        standardize=sidebar_cfg["standardize"],
        p=sidebar_cfg["minkowski_p"],
        max_k=15,
    )

    fig_q1 = plot_training_vs_cv(df_k)
    st.plotly_chart(fig_q1, width="stretch", key="fig_q1_plot")

    train_k1 = float(df_k.loc[df_k["k"] == 1]["train_acc"].iloc[0]) * 100
    cv_k1 = float(df_k.loc[df_k["k"] == 1]["cv_mean"].iloc[0]) * 100
    best_row = df_k.loc[df_k["cv_mean"].idxmax()]

    st.markdown(
        f"""
        <div class="info-card">
            <b>Live Evidence from {bundle.name}:</b><br>
            At <b>K = 1</b>, training accuracy reaches <b>{train_k1:.1f}%</b> while cross-validation accuracy is <b>{cv_k1:.1f}%</b>.
            This demonstrates classic high variance (overfitting): the model memorizes each point's exact label without smoothing noise.
            As K increases to <b>K = {int(best_row['k'])}</b>, cross-validation accuracy peaks at <b>{best_row['cv_mean'] * 100:.2f}%</b>.
            When K is increased further toward 15, decision boundaries become overly smoothed (high bias / underfitting).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Q2
    st.markdown("---")
    st.markdown("#### Q2: Does standardizing features improve KNN accuracy?")

    bundle_raw = get_cached_dataset(bundle.name)
    X_tr, X_te, y_tr, y_te = split_data(bundle_raw.X, bundle_raw.y, test_size=sidebar_cfg["test_size"], random_state=sidebar_cfg["random_seed"])

    std_res = run_standardization_experiment(
        X_tr,
        y_tr,
        X_te,
        y_te,
        bundle.class_names,
        k=sidebar_cfg["manual_k"],
        metric=sidebar_cfg["distance_metric"],
        weights=sidebar_cfg["weights"],
        random_state=sidebar_cfg["random_seed"],
    )

    unscaled_acc = std_res["unscaled"]["test_metrics"]["accuracy"] * 100
    scaled_acc = std_res["scaled"]["test_metrics"]["accuracy"] * 100
    std_delta = scaled_acc - unscaled_acc

    c_s1, c_s2, c_s3 = st.columns(3)
    c_s1.metric("Unstandardized Test Accuracy", f"{unscaled_acc:.2f}%")
    c_s2.metric("Standardized Test Accuracy", f"{scaled_acc:.2f}%")
    c_s3.metric("Standardization Delta", f"{std_delta:+.2f}%")

    st.markdown(
        f"""
        <div class="info-card">
            <b>Live Evidence from {bundle.name}:</b><br>
            Standardization yielded a delta of <b>{std_delta:+.2f} percentage points</b> on the held-out test set.
            Because distance calculations (such as $\\sqrt{{\\sum (x_i - y_i)^2}}$) square numerical differences,
            features with large raw numerical scales disproportionately dominate the geometry.
            StandardScaler normalizes each dimension to mean 0 and variance 1, restoring equal geometric importance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Q3
    st.markdown("---")
    st.markdown("#### Q3: Does tuning K and distance metric together improve performance?")

    if optimized_eval is not None and optimized_params is not None:
        base_acc = default_eval["test_metrics"]["accuracy"] * 100
        opt_acc = optimized_eval["test_metrics"]["accuracy"] * 100
        gain = opt_acc - base_acc

        col_q3_1, col_q3_2, col_q3_3 = st.columns(3)
        col_q3_1.metric("Baseline KNN (K=5, Euclidean)", f"{base_acc:.2f}%")
        col_q3_2.metric(f"Optimized KNN ({optimized_params['metric'].capitalize()}, K={optimized_params['n_neighbors']})", f"{opt_acc:.2f}%")
        col_q3_3.metric("Net Accuracy Gain", f"{gain:+.2f}%")

        st.markdown(
            f"""
            <div class="info-card">
                <b>Live Evidence from {bundle.name}:</b><br>
                Jointly tuning K, distance metric, neighbor weights, and feature scaling using 5-Fold Stratified Cross-Validation
                shifted test performance from <b>{base_acc:.2f}%</b> to <b>{opt_acc:.2f}%</b> (delta: <b>{gain:+.2f}%</b>).
                The optimal combination selected was <b>K = {optimized_params['n_neighbors']}</b>,
                metric = <b>'{optimized_params['metric']}'</b>, weights = <b>'{optimized_params['weights']}'</b>,
                and scaler = <b>'{optimized_params['scaler']}'</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Run hyperparameter optimization in the sidebar to view the empirical joint tuning gain.")

    # Q4
    st.markdown("---")
    st.markdown("#### Q4: Does the best K change across datasets?")

    bench_data = get_cached_multi_dataset_benchmark(
        test_size=sidebar_cfg["test_size"],
        seed=sidebar_cfg["random_seed"],
    )
    df_b = pd.DataFrame(bench_data)

    k_items = [f"**{row['Dataset']}**: Best K = {row['Best K']} ({row['Best Distance Metric']})" for _, row in df_b.iterrows()]
    unique_ks = df_b["Best K"].nunique()

    st.markdown(
        f"""
        <div class="info-card">
            <b>Live Evidence Across All 4 Benchmark Datasets:</b><br>
            {'; '.join(k_items)}.<br><br>
            {'<b>Yes, optimal K differs significantly across datasets.</b> ' if unique_ks > 1 else 'On this partition, optimal K values converged closely, but optimal distance geometries differed. '}
            Because each dataset has distinct sample densities, feature dimensionality, and noise profiles,
            hyperparameters cannot be generalized and must be tuned empirically per domain.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main() -> None:
    """Main application lifecycle and state coordinator."""
    # Ensure session state variables
    if "optimization_completed" not in st.session_state:
        st.session_state["optimization_completed"] = False
        st.session_state["optimized_params"] = None
        st.session_state["optimized_eval"] = None
        st.session_state["optimized_cv"] = None
        st.session_state["current_dataset"] = "Iris"

    # Pre-load initial dataset for sidebar configuration
    current_ds = st.session_state["current_dataset"]
    bundle = get_cached_dataset(current_ds)

    # Render Sidebar
    sidebar_cfg = render_sidebar(bundle)

    # Handle dataset change
    if sidebar_cfg["dataset_name"] != current_ds:
        st.session_state["current_dataset"] = sidebar_cfg["dataset_name"]
        st.session_state["optimization_completed"] = False
        st.session_state["optimized_params"] = None
        st.session_state["optimized_eval"] = None
        st.session_state["optimized_cv"] = None
        bundle = get_cached_dataset(sidebar_cfg["dataset_name"])
        st.rerun()

    # Split dataset into stratified train and held-out test partitions
    X_train, X_test, y_train, y_test = split_data(
        bundle.X,
        bundle.y,
        test_size=sidebar_cfg["test_size"],
        random_state=sidebar_cfg["random_seed"],
    )

    # Always evaluate baseline model (K=5, Euclidean, uniform, unscaled)
    default_eval = run_baseline_model(
        X_train,
        y_train,
        X_test,
        y_test,
        bundle.class_names,
        random_state=sidebar_cfg["random_seed"],
    )

    # Evaluate manual model configured by user in sidebar
    manual_pipe = build_pipeline(
        n_neighbors=sidebar_cfg["manual_k"],
        metric=sidebar_cfg["distance_metric"],
        weights=sidebar_cfg["weights"],
        standardize=sidebar_cfg["standardize"],
        p=sidebar_cfg["minkowski_p"],
    )
    manual_eval = evaluate_pipeline(
        manual_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        bundle.class_names,
        cv_folds=5,
        random_state=sidebar_cfg["random_seed"],
    )

    # Handle Run Optimization Button Click
    if sidebar_cfg["run_opt"]:
        with st.spinner(f"Executing 5-Fold Stratified Cross-Validation search across K=1..15 and metrics on {bundle.name}..."):
            best_params, best_cv, opt_eval = get_cached_grid_search(
                bundle.name,
                sidebar_cfg["test_size"],
                sidebar_cfg["random_seed"],
                max_k=15,
            )
            st.session_state["optimization_completed"] = True
            st.session_state["optimized_params"] = best_params
            st.session_state["optimized_eval"] = opt_eval
            st.session_state["optimized_cv"] = best_cv
            st.session_state["active_k"] = best_params["n_neighbors"]
            st.session_state["best_param_str"] = f"K={best_params['n_neighbors']}, {best_params['metric'].capitalize()}"

    # Determine status and active model
    is_opt = st.session_state["optimization_completed"]
    opt_params = st.session_state["optimized_params"]
    opt_eval = st.session_state["optimized_eval"]
    opt_cv = st.session_state["optimized_cv"]

    status_state = "MODEL_TRAINED" if is_opt else "MODEL_READY"

    # Render Header
    render_header(status_state)

    # Render Top Performance Cards
    render_performance_cards(
        bundle=bundle,
        sidebar_cfg=sidebar_cfg,
        manual_eval=manual_eval,
        optimized_params=opt_params,
        optimized_eval=opt_eval,
        optimized_cv=opt_cv,
    )

    # Select active evaluation and pipeline for prediction and exports
    active_eval = opt_eval if is_opt and opt_eval is not None else manual_eval
    active_pipeline = active_eval["fitted_pipeline"]
    active_k = opt_params["n_neighbors"] if is_opt and opt_params else sidebar_cfg["manual_k"]

    # Streamlit Navigation Tabs
    (
        tab_dash,
        tab_opt,
        tab_bound,
        tab_cm,
        tab_comp,
        tab_pred,
        tab_ds,
        tab_res,
    ) = st.tabs([
        "Dashboard",
        "Optimization",
        "Decision Boundary",
        "Confusion Matrix",
        "Model Comparison",
        "Prediction",
        "Dataset Information",
        "Research & Theory",
    ])

    with tab_dash:
        render_dashboard_tab(
            bundle=bundle,
            sidebar_cfg=sidebar_cfg,
            manual_eval=manual_eval,
            optimized_params=opt_params,
            optimized_eval=opt_eval,
            optimized_cv=opt_cv,
        )

    with tab_opt:
        render_optimization_tab(
            bundle=bundle,
            sidebar_cfg=sidebar_cfg,
            optimized_params=opt_params,
        )

    with tab_bound:
        render_decision_boundary_tab(
            bundle=bundle,
            sidebar_cfg=sidebar_cfg,
        )

    with tab_cm:
        render_confusion_matrix_tab(
            bundle=bundle,
            evaluation=active_eval,
            model_label="Optimized Model" if is_opt else "Manual Model",
        )

    with tab_comp:
        render_comparison_tab(
            bundle=bundle,
            sidebar_cfg=sidebar_cfg,
            default_eval=default_eval,
            optimized_eval=opt_eval,
            optimized_params=opt_params,
            manual_eval=manual_eval,
        )

    with tab_pred:
        render_prediction_tab(
            bundle=bundle,
            pipeline=active_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            k=active_k,
        )

    with tab_ds:
        render_dataset_tab(bundle=bundle)

    with tab_res:
        render_research_tab(
            bundle=bundle,
            sidebar_cfg=sidebar_cfg,
            default_eval=default_eval,
            optimized_eval=opt_eval,
            optimized_params=opt_params,
        )

    # -------------------------------------------------------------------------
    # DOWNLOADS / EXPORT SECTION
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 💾 Export Computed Artifacts")

    d_col1, d_col2, d_col3 = st.columns(3)

    summary_df = export_results_dataframe(
        dataset_name=bundle.name,
        n_samples=bundle.X.shape[0],
        n_features=bundle.X.shape[1],
        n_classes=len(bundle.class_names),
        model_type="Optimized" if is_opt else "Manual",
        k=active_k,
        metric=opt_params["metric"] if is_opt and opt_params else sidebar_cfg["distance_metric"],
        weights=opt_params["weights"] if is_opt and opt_params else sidebar_cfg["weights"],
        scaling=opt_params["scaler"] if is_opt and opt_params else ("StandardScaler" if sidebar_cfg["standardize"] else "passthrough"),
        cv_accuracy=opt_cv if is_opt and opt_cv is not None else manual_eval["cv_mean"],
        test_accuracy=active_eval["test_metrics"]["accuracy"],
        macro_precision=active_eval["test_metrics"]["macro_precision"],
        macro_recall=active_eval["test_metrics"]["macro_recall"],
        macro_f1=active_eval["test_metrics"]["macro_f1"],
    )
    csv_summary = summary_df.to_csv(index=False).encode("utf-8")

    with d_col1:
        st.download_button(
            label="📥 Download Results Summary CSV",
            data=csv_summary,
            file_name=f"{bundle.name.lower().replace(' ', '_')}_model_results.csv",
            mime="text/csv",
            width="stretch",
            key="btn_download_summary_csv",
        )

    with d_col2:
        csv_report = active_eval["report_df"].to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Classification Report CSV",
            data=csv_report,
            file_name=f"{bundle.name.lower().replace(' ', '_')}_classification_report.csv",
            mime="text/csv",
            width="stretch",
            key="btn_download_cm_report",
        )

    with d_col3:
        if is_opt and opt_params:
            json_str = serialize_params(opt_params)
        else:
            json_str = serialize_params({
                "n_neighbors": sidebar_cfg["manual_k"],
                "metric": sidebar_cfg["distance_metric"],
                "minkowski_p": sidebar_cfg["minkowski_p"],
                "weights": sidebar_cfg["weights"],
                "scaler": "StandardScaler" if sidebar_cfg["standardize"] else "passthrough",
            })

        st.download_button(
            label="📥 Download Parameters JSON",
            data=json_str.encode("utf-8"),
            file_name=f"{bundle.name.lower().replace(' ', '_')}_parameters.json",
            mime="application/json",
            width="stretch",
            key="btn_download_params_json",
        )

    # Footer
    st.markdown(
        """
        <div class="app-footer">
            SMART ML PARAMETER TUNER &mdash; AutoML Mini Lab &bull; Interactive KNN Classification & Hyperparameter Optimization
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
