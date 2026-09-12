from ml.model_registry import feature_type_for, pick_winner


def _record(task, model_name, split, **kwargs):
    return {"task": task, "model_name": model_name, "split": split, **kwargs}


def test_pick_winner_classification_uses_pr_auc_not_roc_auc():
    records = [
        _record("t", "logistic_regression_fingerprints", "val", roc_auc=0.9, pr_auc=0.3),
        _record("t", "random_forest_descriptors", "val", roc_auc=0.7, pr_auc=0.6),
        _record("t", "majority_class", "val", roc_auc=0.5, pr_auc=0.9),  # excluded, not a real candidate
    ]
    winner = pick_winner(records, "t", is_classification=True)
    assert winner["model_name"] == "random_forest_descriptors"


def test_pick_winner_classification_skips_undefined_pr_auc():
    records = [
        _record("t", "logistic_regression_fingerprints", "val", roc_auc=None, pr_auc=None),
        _record("t", "random_forest_descriptors", "val", roc_auc=0.8, pr_auc=0.4),
    ]
    winner = pick_winner(records, "t", is_classification=True)
    assert winner["model_name"] == "random_forest_descriptors"


def test_pick_winner_classification_no_valid_candidate_returns_none():
    records = [_record("t", "logistic_regression_fingerprints", "val", roc_auc=None, pr_auc=None)]
    assert pick_winner(records, "t", is_classification=True) is None


def test_pick_winner_regression_uses_lowest_rmse():
    records = [
        _record("solubility", "ridge_descriptors", "val", rmse=0.9),
        _record("solubility", "xgboost_tuned", "val", rmse=0.7),
        _record("solubility", "naive_mean", "val", rmse=1.5),  # excluded, not a real candidate
    ]
    winner = pick_winner(records, "solubility", is_classification=False)
    assert winner["model_name"] == "xgboost_tuned"


def test_pick_winner_ignores_other_tasks_and_splits():
    records = [
        _record("other_task", "xgboost_tuned", "val", rmse=0.1),
        _record("solubility", "xgboost_tuned", "test", rmse=0.2),
        _record("solubility", "ridge_descriptors", "val", rmse=0.8),
    ]
    winner = pick_winner(records, "solubility", is_classification=False)
    assert winner["model_name"] == "ridge_descriptors"


def test_feature_type_for_logistic_regression_is_fingerprints():
    assert feature_type_for("logistic_regression_fingerprints") == "fingerprints"


def test_feature_type_for_other_models_is_descriptors():
    assert feature_type_for("random_forest_descriptors") == "descriptors"
    assert feature_type_for("xgboost_tuned") == "descriptors"
