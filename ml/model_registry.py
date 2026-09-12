"""Select a per-task winner (from ml/baseline.py + ml/experiments_per_task_models.py
candidates), retrain each winner on the full train split, calibrate
classification tasks with Platt scaling (found to win 3/4 times in
ml/calibration.py's exploration - applied here to every classification
task's final model, not just the 4 explored), and package all 16 models
into models/production/ - the "one model per task" analogue of project
#1's single-model registry.

Usage:
    python ml/model_registry.py

Reads ml/results/{baseline,per_task_xgboost}_metrics.json + data/processed/admet_processed.csv.
Writes models/production/models.joblib (dict of per-task model bundles)
and models/production/metadata.json (matches backend-spec/api-contract.md's
GET /admet-profile response shape for metrics/limitations).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    REGRESSION_TASKS,
    compute_descriptors,
    compute_fingerprints,
    task_rows,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
BASELINE_METRICS_PATH = ROOT / "ml" / "results" / "baseline_metrics.json"
PER_TASK_METRICS_PATH = ROOT / "ml" / "results" / "per_task_xgboost_metrics.json"
PRODUCTION_DIR = ROOT / "models" / "production"
SEED = 42
MODEL_VERSION = "0.1.0"

# Candidate model_names eligible to win, per task type. "majority_class"/
# "naive_mean" are the floor, not real candidates - excluded from winning.
CLASSIFICATION_CANDIDATES = ["logistic_regression_fingerprints", "random_forest_descriptors", "xgboost_tuned"]
REGRESSION_CANDIDATES = ["ridge_descriptors", "xgboost_tuned"]


def load_all_metrics() -> list[dict]:
    records = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    records += json.loads(PER_TASK_METRICS_PATH.read_text(encoding="utf-8"))
    return records


def pick_winner(records: list[dict], task: str, is_classification: bool) -> dict | None:
    candidates = CLASSIFICATION_CANDIDATES if is_classification else REGRESSION_CANDIDATES
    val_records = [r for r in records if r["task"] == task and r["split"] == "val" and r["model_name"] in candidates]
    if is_classification:
        # PR-AUC, not ROC-AUC - these tasks are imbalanced, PR-AUC is the
        # more informative metric here (see ml/results/per_task_comparison_report.md).
        scored = [r for r in val_records if r.get("pr_auc") is not None]
        if not scored:
            return None
        return max(scored, key=lambda r: r["pr_auc"])
    scored = [r for r in val_records if r.get("rmse") is not None]
    if not scored:
        return None
    return min(scored, key=lambda r: r["rmse"])


def build_classifier(model_name: str, y_train):
    """Returns an UNFITTED estimator - CalibratedClassifierCV(cv=3) fits
    fresh copies internally per fold, so fitting here first would just be
    wasted computation (and slightly misleading, since that pre-fit copy
    is discarded)."""
    if model_name == "logistic_regression_fingerprints":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if model_name == "random_forest_descriptors":
        return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    if model_name == "xgboost_tuned":
        neg, pos = np.bincount(y_train.astype(int))
        return XGBClassifier(random_state=SEED, n_jobs=-1, scale_pos_weight=neg / max(pos, 1), eval_metric="logloss")
    raise ValueError(f"Unknown classification model_name: {model_name}")


def retrain_regressor(model_name: str, X_train, y_train):
    if model_name == "ridge_descriptors":
        scaler = StandardScaler().fit(X_train)
        model = Ridge(alpha=1.0, random_state=SEED).fit(scaler.transform(X_train), y_train)
        return model, scaler
    if model_name == "xgboost_tuned":
        return XGBRegressor(random_state=SEED, n_jobs=-1, max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train), None
    raise ValueError(f"Unknown regression model_name: {model_name}")


def feature_type_for(model_name: str) -> str:
    return "fingerprints" if model_name == "logistic_regression_fingerprints" else "descriptors"


def evaluate_classification(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": float(roc_auc_score(y, proba)), "pr_auc": float(average_precision_score(y, proba))}


def evaluate_regression(model, X, y, scaler=None) -> dict:
    X_in = scaler.transform(X) if scaler is not None else X
    pred = model.predict(X_in)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index
    fingerprints = compute_fingerprints(df["canonical_smiles"])
    all_metrics = load_all_metrics()
    trained_at = datetime.now(timezone.utc).isoformat()

    task_bundles = {}
    task_metadata = {}
    no_winner_tasks = []

    for task in REGRESSION_TASKS + CLASSIFICATION_TASKS:
        is_classification = task in CLASSIFICATION_TASKS
        winner_val = pick_winner(all_metrics, task, is_classification)
        if winner_val is None:
            no_winner_tasks.append(task)
            continue
        model_name = winner_val["model_name"]

        rows = task_rows(df, task)
        row_positions = df.index.get_indexer(rows.index)
        train_mask = (rows["split"] == "train").to_numpy()
        desc = descriptors.loc[rows.index, DESCRIPTOR_COLUMNS].to_numpy()
        fps = fingerprints[row_positions]

        if is_classification:
            y = rows[task].to_numpy().astype(int)
            X = fps if feature_type_for(model_name) == "fingerprints" else desc
            if len(np.unique(y[train_mask])) < 2:
                no_winner_tasks.append(task)
                continue
            base_model = build_classifier(model_name, y[train_mask])
            # Platt scaling: won 3/4 tasks in ml/calibration.py's exploration,
            # applied here as the default for every classification task's
            # production model (not re-litigated per task - isotonic never
            # won there and sometimes did worse than uncalibrated on small data).
            # cv=3 (not "prefit"): fits fresh copies of base_model on each
            # training fold internally and calibrates on the held-out fold -
            # the standard sklearn pattern when you have enough data to spare.
            # A handful of tasks have very few positives in train, where a
            # 3-way stratified split can still leave a fold degenerate -
            # fall back to 2-fold rather than dropping the task entirely.
            n_minority_train = int(min(np.bincount(y[train_mask])))
            cv_folds = 3 if n_minority_train >= 6 else 2
            calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=cv_folds).fit(X[train_mask], y[train_mask])

            test_mask = (rows["split"] == "test").to_numpy()
            val_metrics = evaluate_classification(calibrated, X[(rows["split"] == "val").to_numpy()], y[(rows["split"] == "val").to_numpy()])
            test_metrics = evaluate_classification(calibrated, X[test_mask], y[test_mask]) if test_mask.sum() else {"roc_auc": None, "pr_auc": None}

            task_bundles[task] = {
                "model": calibrated, "model_name": model_name, "task_type": "classification",
                "feature_type": feature_type_for(model_name), "calibrated": True,
            }
            task_metadata[task] = {"model_name": model_name, "val": val_metrics, "test": test_metrics}
        else:
            y = rows[task].to_numpy()
            X = desc
            model, scaler = retrain_regressor(model_name, X[train_mask], y[train_mask])

            test_mask = (rows["split"] == "test").to_numpy()
            val_mask = (rows["split"] == "val").to_numpy()
            val_metrics = evaluate_regression(model, X[val_mask], y[val_mask], scaler)
            test_metrics = evaluate_regression(model, X[test_mask], y[test_mask], scaler) if test_mask.sum() else {}

            task_bundles[task] = {
                "model": model, "model_name": model_name, "task_type": "regression",
                "feature_type": "descriptors", "calibrated": False, "scaler": scaler,
            }
            task_metadata[task] = {"model_name": model_name, "val": val_metrics, "test": test_metrics}

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tasks": task_bundles, "descriptor_columns": DESCRIPTOR_COLUMNS}, PRODUCTION_DIR / "models.joblib")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "dataset": "ESOL + Tox21 + ClinTox + BBBP (MoleculeNet)",
        "per_task_winners": task_metadata,
        "tasks_without_a_registered_model": no_winner_tasks,
        "known_limitations": [
            (
                "Per-task models, not a true multi-task architecture - each task's "
                "winner is trained independently (see ml/TODO_experiments_multitask.md "
                "for the planned multi-task comparison, not yet done)."
            ),
            (
                "Calibrated via Platt scaling (sigmoid) for every classification task - "
                "chosen because it won 3/4 tasks explored in ml/results/calibration_report.md; "
                "not re-verified per-task here, so a specific task's calibration quality may "
                "differ from those 4 explored cases."
            ),
            (
                "These 4 source datasets barely overlap by molecule - most training "
                "examples for any one task come from a single source dataset, so "
                "cross-task generalization claims should not be assumed."
            ),
        ],
    }
    if no_winner_tasks:
        metadata["known_limitations"].append(
            f"No model could be registered for: {', '.join(no_winner_tasks)} "
            "(train split had only one class present for that task)."
        )
    (PRODUCTION_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Registered {len(task_bundles)} / {len(REGRESSION_TASKS + CLASSIFICATION_TASKS)} tasks.")
    if no_winner_tasks:
        print(f"No winner for: {no_winner_tasks}")
    for task, meta in task_metadata.items():
        print(f"  {task}: {meta['model_name']} -> val {meta['val']}")
    print(f"\nWrote {PRODUCTION_DIR / 'models.joblib'}\nWrote {PRODUCTION_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
