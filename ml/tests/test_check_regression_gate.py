from ml.check_regression_gate import (
    _val_metric_and_direction,
    check_per_task_metrics,
    compare_to_historical_metadata,
)

THRESHOLDS = {"tolerance": {"classification_pr_auc": 0.035, "regression_rmse": 0.02, "calibration_ece": 0.05}}


def test_val_metric_and_direction_classification():
    metric, value, higher_is_better = _val_metric_and_direction({"val": {"pr_auc": 0.5, "roc_auc": 0.7}})
    assert metric == "pr_auc"
    assert value == 0.5
    assert higher_is_better is True


def test_val_metric_and_direction_regression():
    metric, value, higher_is_better = _val_metric_and_direction({"val": {"rmse": 0.7, "mae": 0.5}})
    assert metric == "rmse"
    assert value == 0.7
    assert higher_is_better is False


def test_val_metric_and_direction_neither_key_returns_none():
    assert _val_metric_and_direction({"val": {}}) is None


def test_compare_to_historical_metadata_pass_within_tolerance():
    historical = {"per_task_winners": {"NR-AR": {"val": {"pr_auc": 0.50}}}}
    current = {"per_task_winners": {"NR-AR": {"val": {"pr_auc": 0.49}}}}  # tiny drop, within 0.035 tol

    results = compare_to_historical_metadata(current, historical, THRESHOLDS, "abc123")

    assert len(results) == 1
    assert results[0]["ok"] is True


def test_compare_to_historical_metadata_fails_on_real_regression():
    historical = {"per_task_winners": {"NR-AR": {"val": {"pr_auc": 0.50}}}}
    current = {"per_task_winners": {"NR-AR": {"val": {"pr_auc": 0.30}}}}  # well past tolerance

    results = compare_to_historical_metadata(current, historical, THRESHOLDS, "abc123")

    assert results[0]["ok"] is False
    assert "NR-AR" in results[0]["task"]


def test_compare_to_historical_metadata_regression_rmse_direction():
    historical = {"per_task_winners": {"solubility": {"val": {"rmse": 0.70}}}}
    current_worse = {"per_task_winners": {"solubility": {"val": {"rmse": 0.95}}}}  # RMSE got much worse (higher)

    results = compare_to_historical_metadata(current_worse, historical, THRESHOLDS, "abc123")

    assert results[0]["ok"] is False


def test_compare_to_historical_metadata_task_no_longer_registered_is_a_regression():
    historical = {"per_task_winners": {"NR-AR": {"val": {"pr_auc": 0.50}}}}
    current = {"per_task_winners": {}}  # task dropped entirely from current registry

    results = compare_to_historical_metadata(current, historical, THRESHOLDS, "abc123")

    assert results[0]["ok"] is False
    assert "no longer registered" in results[0]["reason"]


def test_check_per_task_metrics_still_works_unchanged():
    # Regression guard: the pre-existing default-mode gate logic must be untouched.
    metadata = {"per_task_winners": {"solubility": {"val": {"rmse": 0.70}}}}
    thresholds = {
        **THRESHOLDS,
        "per_task": {"solubility": {"metric": "rmse", "direction": "lower_is_better", "value": 0.70}},
    }
    results = check_per_task_metrics(metadata, thresholds)
    assert results[0]["ok"] is True
