"""
Integration and unit tests for SMART ML PARAMETER TUNER.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src


def test_pipeline_all_datasets():
    print("1. Testing data loading...")
    for name in ["Iris", "Wine", "Breast Cancer", "Digits"]:
        b = src.load_dataset(name)
        assert b.X.shape[0] > 0
        assert b.X.shape[1] > 0
        assert len(b.class_names) >= 2
        summary = src.get_dataset_summary(b)
        assert summary["n_samples"] == b.X.shape[0]
        print(f"   Loaded {b.name}: {b.X.shape}, {len(b.class_names)} classes - OK")

    print("2. Testing preprocessing & splitting...")
    b = src.load_dataset("Iris")
    X_tr, X_te, y_tr, y_te = src.split_data(b.X, b.y, test_size=0.2, random_state=42)
    assert len(X_tr) == 120
    assert len(X_te) == 30
    print("   Split data - OK")

    print("3. Testing compute_accuracy_vs_k...")
    df_k = src.compute_accuracy_vs_k(X_tr, y_tr, metric="euclidean", weights="uniform", standardize=True, max_k=15, random_state=42)
    assert len(df_k) == 15
    assert "cv_mean" in df_k.columns
    assert "train_acc" in df_k.columns
    best_k = int(df_k.loc[df_k["cv_mean"].idxmax()]["k"])
    print(f"   Accuracy vs K computed: shape {df_k.shape}, best K: {best_k} - OK")

    print("4. Testing grid search...")
    grid, best_p = src.run_grid_search(X_tr, y_tr, max_k=5, random_state=42)
    assert "n_neighbors" in best_p
    assert "metric" in best_p
    print(f"   Grid search best params: {best_p} - OK")

    print("5. Testing evaluation pipeline...")
    pipe = src.build_pipeline(n_neighbors=5, metric="euclidean", weights="uniform", standardize=True)
    res = src.evaluate_pipeline(pipe, X_tr, y_tr, X_te, y_te, b.class_names, random_state=42)
    assert "test_metrics" in res
    assert 0.0 <= res["test_metrics"]["accuracy"] <= 1.0
    print(f"   Test accuracy: {res['test_metrics']['accuracy']:.4f} - OK")

    print("6. Testing Plotly visualizations...")
    fig1 = src.plot_accuracy_vs_k(df_k, best_k=best_k)
    fig2 = src.plot_confusion_matrix(res["confusion_matrix"], b.class_names)
    assert fig1 is not None
    assert fig2 is not None
    print("   Visualizations - OK")

    print("7. Testing 2D Decision Boundary...")
    pipe_2d = src.train_2d_model(
        X_tr.values[:, :2],
        y_tr.values,
        feature_names=b.feature_names[:2],
        n_neighbors=5,
    )
    fig3 = src.plot_decision_boundary(
        pipe_2d,
        X_tr.values[:, :2],
        y_tr.values,
        X_te.values[:, :2],
        y_te.values,
        b.feature_names[:2],
        b.class_names,
    )
    assert fig3 is not None
    print("   Decision Boundary - OK")

    print("8. Testing Neighbor Retrieval...")
    neighbors = src.get_k_nearest_neighbors(pipe, X_tr, y_tr, X_te.values[0], n_neighbors=5, class_names=b.class_names)
    assert len(neighbors) == 5
    print(f"   Neighbors retrieved: {len(neighbors)} - OK")

    print("9. Testing Standardization Experiment...")
    std_res = src.run_standardization_experiment(X_tr, y_tr, X_te, y_te, b.class_names, random_state=42)
    assert "unscaled" in std_res and "scaled" in std_res
    print(f"   Standardization experiment: unscaled={std_res['unscaled']['test_metrics']['accuracy']:.3f}, scaled={std_res['scaled']['test_metrics']['accuracy']:.3f} - OK")

    print("\n>>> ALL TESTS COMPLETED AND PASSED! <<<")


if __name__ == "__main__":
    test_pipeline_all_datasets()
