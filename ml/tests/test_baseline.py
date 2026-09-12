import numpy as np

from ml.baseline import classification_metrics, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_classification_metrics_normal_case():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y_true, y_proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["n_positive"] == 2
    assert metrics["n_total"] == 4


def test_classification_metrics_single_class_returns_none_not_error():
    # A rare task can end up with only one class in a small val/test split -
    # this must not raise (sklearn's roc_auc_score would).
    y_true = np.array([0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3])
    metrics = classification_metrics(y_true, y_proba)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["n_positive"] == 0
    assert metrics["n_total"] == 3
