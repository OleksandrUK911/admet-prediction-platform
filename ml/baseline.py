"""Per-task baseline models for all 16 ADMET targets (1 regression +
15 classification), per ml/TODO_baseline.md.

Usage:
    python ml/baseline.py

For each task, only rows with a non-missing label are used (masking, not
imputation). Models per task type:
  - Regression (solubility): naive mean, Ridge on descriptors
  - Classification (15 tasks): majority-class, Logistic Regression on
    Morgan fingerprints, Random Forest on descriptors

Writes ml/results/baseline_metrics.json (one record per task/model/split)
and ml/results/baseline_comparison.md.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
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
RESULTS_PATH = ROOT / "ml" / "results" / "baseline_metrics.json"
REPORT_PATH = ROOT / "ml" / "results" / "baseline_comparison.md"
SEED = 42


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_proba) -> dict:
    """ROC-AUC/PR-AUC are undefined with only one class present (small/rare
    tasks can hit this on a small val/test split) - return None rather than
    letting sklearn raise or silently produce a misleading number."""
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None, "n_positive": int(np.sum(y_true)), "n_total": len(y_true)}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "n_positive": int(np.sum(y_true)),
        "n_total": len(y_true),
    }


def run_regression_task(task: str, df: pd.DataFrame, descriptors: pd.DataFrame, trained_at: str) -> list[dict]:
    rows = task_rows(df, task)
    desc = descriptors.loc[rows.index]
    train_mask = (rows["split"] == "train").to_numpy()
    records = []

    y = rows[task].to_numpy()
    X = desc.to_numpy()

    naive = DummyRegressor(strategy="mean").fit(X[train_mask], y[train_mask])
    scaler = StandardScaler().fit(X[train_mask])
    ridge = Ridge(alpha=1.0, random_state=SEED).fit(scaler.transform(X[train_mask]), y[train_mask])

    for split in ["val", "test"]:
        mask = (rows["split"] == split).to_numpy()
        if mask.sum() == 0:
            continue
        records.append({
            "task": task, "model_name": "naive_mean", "split": split, "trained_at": trained_at,
            "n_samples": int(mask.sum()), **regression_metrics(y[mask], naive.predict(X[mask])),
        })
        records.append({
            "task": task, "model_name": "ridge_descriptors", "split": split, "trained_at": trained_at,
            "n_samples": int(mask.sum()), **regression_metrics(y[mask], ridge.predict(scaler.transform(X[mask]))),
        })
    return records


def run_classification_task(
    task: str, df: pd.DataFrame, descriptors: pd.DataFrame, fingerprints: np.ndarray, trained_at: str
) -> list[dict]:
    rows = task_rows(df, task)
    row_positions = df.index.get_indexer(rows.index)
    desc = descriptors.loc[rows.index].to_numpy()
    fps = fingerprints[row_positions]
    y = rows[task].to_numpy().astype(int)
    train_mask = (rows["split"] == "train").to_numpy()

    records = []
    if len(np.unique(y[train_mask])) < 2:
        # Can't train a classifier with only one class in train for this task.
        return records

    majority = DummyClassifier(strategy="most_frequent").fit(fps[train_mask], y[train_mask])
    lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(fps[train_mask], y[train_mask])
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(desc[train_mask], y[train_mask])

    for split in ["val", "test"]:
        mask = (rows["split"] == split).to_numpy()
        if mask.sum() == 0:
            continue
        y_split = y[mask]

        majority_proba = majority.predict_proba(fps[mask])
        majority_score = majority_proba[:, 1] if majority_proba.shape[1] > 1 else np.zeros(mask.sum())
        records.append({
            "task": task, "model_name": "majority_class", "split": split, "trained_at": trained_at,
            **classification_metrics(y_split, majority_score),
        })
        records.append({
            "task": task, "model_name": "logistic_regression_fingerprints", "split": split, "trained_at": trained_at,
            **classification_metrics(y_split, lr.predict_proba(fps[mask])[:, 1]),
        })
        records.append({
            "task": task, "model_name": "random_forest_descriptors", "split": split, "trained_at": trained_at,
            **classification_metrics(y_split, rf.predict_proba(desc[mask])[:, 1]),
        })
    return records


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index
    fingerprints = compute_fingerprints(df["canonical_smiles"])
    trained_at = datetime.now(timezone.utc).isoformat()

    all_records = []
    for task in REGRESSION_TASKS:
        all_records.extend(run_regression_task(task, df, descriptors[DESCRIPTOR_COLUMNS], trained_at))
    for task in CLASSIFICATION_TASKS:
        all_records.extend(run_classification_task(task, df, descriptors[DESCRIPTOR_COLUMNS], fingerprints, trained_at))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_records, indent=2), encoding="utf-8")

    lines = ["# ADMET baseline comparison\n\n"]
    lines.append("## Regression (solubility)\n\n| Task | Model | Split | N | RMSE | MAE | R2 |\n|---|---|---|---|---|---|---|\n")
    for r in [r for r in all_records if r["task"] in REGRESSION_TASKS]:
        lines.append(f"| {r['task']} | {r['model_name']} | {r['split']} | {r['n_samples']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} |\n")

    lines.append("\n## Classification (15 tasks)\n\n| Task | Model | Split | N | N+ | ROC-AUC | PR-AUC |\n|---|---|---|---|---|---|---|\n")
    for r in [r for r in all_records if r["task"] in CLASSIFICATION_TASKS]:
        roc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "n/a (1 class)"
        pr = f"{r['pr_auc']:.3f}" if r["pr_auc"] is not None else "n/a"
        lines.append(f"| {r['task']} | {r['model_name']} | {r['split']} | {r['n_total']} | {r['n_positive']} | {roc} | {pr} |\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {len(all_records)} baseline records.")
    print(f"Wrote {RESULTS_PATH}\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
