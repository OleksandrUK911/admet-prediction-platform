from ml.evaluate import (
    close_enough,
    diagnose_worst_classification_tasks,
    overfitting_flags,
)


def test_close_enough_within_tolerance():
    assert close_enough(0.50001, 0.5, tol=1e-3)


def test_close_enough_outside_tolerance():
    assert not close_enough(0.6, 0.5, tol=1e-3)


def test_close_enough_both_none():
    assert close_enough(None, None)


def test_close_enough_one_none():
    assert not close_enough(None, 0.5)


def _classification_info(train_pr, val_pr):
    return {"task_type": "classification", "train": {"pr_auc": train_pr}, "val": {"pr_auc": val_pr}}


def _regression_info(train_rmse, val_rmse):
    return {"task_type": "regression", "train": {"rmse": train_rmse}, "val": {"rmse": val_rmse}}


def test_overfitting_flags_detects_large_classification_gap():
    per_task = {"t": _classification_info(train_pr=0.95, val_pr=0.5)}
    flags = overfitting_flags(per_task)
    assert len(flags) == 1
    assert flags[0]["task"] == "t"
    assert flags[0]["metric"] == "pr_auc"


def test_overfitting_flags_ignores_small_classification_gap():
    per_task = {"t": _classification_info(train_pr=0.55, val_pr=0.5)}
    assert overfitting_flags(per_task) == []


def test_overfitting_flags_detects_large_regression_gap_ratio():
    per_task = {"t": _regression_info(train_rmse=0.1, val_rmse=0.9)}
    flags = overfitting_flags(per_task)
    assert len(flags) == 1
    assert flags[0]["metric"] == "rmse"


def test_overfitting_flags_ignores_small_regression_gap_ratio():
    per_task = {"t": _regression_info(train_rmse=0.8, val_rmse=0.9)}
    assert overfitting_flags(per_task) == []


def test_diagnose_worst_classification_tasks_ranks_lowest_pr_auc_first():
    per_task = {
        "good": {"task_type": "classification", "test": {"pr_auc": 0.9}, "n_positive_total": 50, "n_train": 60, "n_val": 20, "n_test": 20},
        "bad": {"task_type": "classification", "test": {"pr_auc": 0.1}, "n_positive_total": 5, "n_train": 80, "n_val": 10, "n_test": 10},
        "regression_task": {"task_type": "regression", "test": {"r2": 0.8}},
    }
    worst = diagnose_worst_classification_tasks(per_task, n=1)
    assert len(worst) == 1
    assert worst[0]["task"] == "bad"
    assert "diagnosis" in worst[0]
