"""Per-task XGBoost models for all 16 ADMET targets (1 regression + 15
classification), trained on the 7 shared descriptors.

Unlike ml/baseline.py, these models train a single XGBoost model per task
(no multi-task learning - that is a separate experiment out of scope here)
and, for the 15 classification tasks, explicitly address class imbalance
via XGBoost's `scale_pos_weight` (ratio of negatives to positives in the
TRAIN split for that task) rather than leaving it at library defaults like
the baseline's RandomForestClassifier does. Imbalanced tasks (e.g. some
Tox21 assays have <2% positives) are better judged by PR-AUC than ROC-AUC:
ROC-AUC's false-positive-rate axis is diluted by the huge number of true
negatives, so a classifier can score well on ROC-AUC while still producing
mostly false alarms among its positive predictions - PR-AUC, which only
looks at precision/recall over the (rare) positive class, is more sensitive
to that failure mode and is the more honest metric for these tasks.

For the regression task (solubility), a small fixed grid of XGBoost
hyperparameter configs is tried and the one with the best val RMSE is kept
(same train-fit / val-select / test-report-only discipline used elsewhere
in this project).

Usage:
    python ml/experiments_per_task_models.py

Writes ml/results/per_task_xgboost_metrics.json (schema matching
ml/results/baseline_metrics.json's family: one record per task/split) and
ml/results/per_task_comparison_report.md (baseline-best vs. this model,
per task, on val and test).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    REGRESSION_TASKS,
    compute_descriptors,
    task_rows,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
BASELINE_METRICS_PATH = ROOT / "ml" / "results" / "baseline_metrics.json"
RESULTS_PATH = ROOT / "ml" / "results" / "per_task_xgboost_metrics.json"
REPORT_PATH = ROOT / "ml" / "results" / "per_task_comparison_report.md"
SEED = 42
MODEL_NAME = "xgboost_tuned"

# Small, cheap fixed grid for the regression task's hyperparameter search.
REGRESSION_GRID = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03},
]


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_proba) -> dict:
    """ROC-AUC/PR-AUC are undefined with only one class present - return
    None rather than letting sklearn raise or produce a misleading number."""
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
    y = rows[task].to_numpy()
    X = desc.to_numpy()

    train_mask = (rows["split"] == "train").to_numpy()
    val_mask = (rows["split"] == "val").to_numpy()
    test_mask = (rows["split"] == "test").to_numpy()

    best_model = None
    best_config = None
    best_val_rmse = np.inf
    for config in REGRESSION_GRID:
        model = XGBRegressor(
            **config,
            random_state=SEED,
            objective="reg:squarederror",
            n_jobs=-1,
        ).fit(X[train_mask], y[train_mask])
        val_rmse = regression_metrics(y[val_mask], model.predict(X[val_mask]))["rmse"]
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_config = config

    print(f"  [{task}] best config: {best_config} (val RMSE={best_val_rmse:.4f})")

    records = []
    for split, mask in [("val", val_mask), ("test", test_mask)]:
        if mask.sum() == 0:
            continue
        records.append({
            "task": task, "model_name": MODEL_NAME, "split": split, "trained_at": trained_at,
            "n_samples": int(mask.sum()), "best_params": best_config,
            **regression_metrics(y[mask], best_model.predict(X[mask])),
        })
    return records


def run_classification_task(task: str, df: pd.DataFrame, descriptors: pd.DataFrame, trained_at: str) -> list[dict]:
    rows = task_rows(df, task)
    desc = descriptors.loc[rows.index].to_numpy()
    y = rows[task].to_numpy().astype(int)
    train_mask = (rows["split"] == "train").to_numpy()

    if len(np.unique(y[train_mask])) < 2:
        # Can't train a classifier with only one class in train for this task.
        return []

    n_pos_train = int(np.sum(y[train_mask] == 1))
    n_neg_train = int(np.sum(y[train_mask] == 0))
    scale_pos_weight = n_neg_train / n_pos_train

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    ).fit(desc[train_mask], y[train_mask])

    records = []
    for split in ["val", "test"]:
        mask = (rows["split"] == split).to_numpy()
        if mask.sum() == 0:
            continue
        y_split = y[mask]
        proba = model.predict_proba(desc[mask])[:, 1]
        records.append({
            "task": task, "model_name": MODEL_NAME, "split": split, "trained_at": trained_at,
            "scale_pos_weight": scale_pos_weight,
            **classification_metrics(y_split, proba),
        })
    return records


def _load_baseline_best(baseline_records: list[dict]) -> dict:
    """For each classification task, pick whichever baseline model had the
    best val ROC-AUC (falling back to PR-AUC comparison isn't needed since
    ROC-AUC and PR-AUC track each other closely enough here; ROC-AUC is what
    ml/baseline.py's models were compared on). Returns
    {task: {"model_name": ..., split: {roc_auc, pr_auc, n_positive, n_total}}}.
    """
    by_task_model = {}
    for r in baseline_records:
        if r["task"] not in CLASSIFICATION_TASKS:
            continue
        by_task_model.setdefault(r["task"], {}).setdefault(r["model_name"], {})[r["split"]] = r

    best = {}
    for task, models in by_task_model.items():
        best_model_name, best_val_roc = None, -np.inf
        for model_name, splits in models.items():
            val = splits.get("val")
            if val is None or val.get("roc_auc") is None:
                continue
            if val["roc_auc"] > best_val_roc:
                best_val_roc = val["roc_auc"]
                best_model_name = model_name
        if best_model_name is None:
            # All models had undefined ROC-AUC on val (single class) - fall
            # back to whichever model exists at all, arbitrarily the first.
            best_model_name = next(iter(models))
        best[task] = {"model_name": best_model_name, **models[best_model_name]}
    return best


def _load_baseline_regression(baseline_records: list[dict]) -> dict:
    """Best regression baseline model per task, by val RMSE."""
    by_task_model = {}
    for r in baseline_records:
        if r["task"] not in REGRESSION_TASKS:
            continue
        by_task_model.setdefault(r["task"], {}).setdefault(r["model_name"], {})[r["split"]] = r

    best = {}
    for task, models in by_task_model.items():
        best_model_name, best_val_rmse = None, np.inf
        for model_name, splits in models.items():
            val = splits.get("val")
            if val is None:
                continue
            if val["rmse"] < best_val_rmse:
                best_val_rmse = val["rmse"]
                best_model_name = model_name
        best[task] = {"model_name": best_model_name, **models[best_model_name]}
    return best


def _fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_report(all_records: list[dict], baseline_records: list[dict]) -> None:
    baseline_best_cls = _load_baseline_best(baseline_records)
    baseline_best_reg = _load_baseline_regression(baseline_records)

    xgb_by_task = {}
    for r in all_records:
        xgb_by_task.setdefault(r["task"], {})[r["split"]] = r

    lines = ["# Per-task XGBoost vs. baseline comparison\n\n"]
    lines.append(
        "One XGBoost model per task, trained on the 7 shared descriptors. "
        "Classification tasks use `scale_pos_weight` (train-split "
        "negatives / positives) to address class imbalance, instead of "
        "library defaults. Baseline-best is whichever of "
        "majority/logistic-regression/random-forest had the best val "
        "ROC-AUC (regression: best val RMSE) in `ml/results/baseline_metrics.json`.\n\n"
        "Imbalanced tasks are judged primarily by PR-AUC, not ROC-AUC: "
        "ROC-AUC's false-positive-rate axis is diluted by the large number "
        "of true negatives typical of these tasks, so a classifier can look "
        "good on ROC-AUC while its positive predictions are still mostly "
        "wrong. PR-AUC, which only considers precision/recall over the "
        "(rare) positive class, is the more honest signal for whether "
        "class-weighting actually helped.\n\n"
    )

    lines.append("## Regression (solubility)\n\n")
    lines.append("| Split | Baseline-best | Baseline RMSE | XGBoost RMSE | XGBoost R2 | Improved? |\n|---|---|---|---|---|---|\n")
    # _load_baseline_regression only kept the val record used for model
    # selection; rebuild a per-split lookup here for the val+test rows below.
    reg_by_task_model_split = {}
    for r in baseline_records:
        if r["task"] not in REGRESSION_TASKS:
            continue
        reg_by_task_model_split.setdefault(r["task"], {}).setdefault(r["model_name"], {})[r["split"]] = r

    for task in REGRESSION_TASKS:
        best_name = baseline_best_reg[task]["model_name"]
        for split in ["val", "test"]:
            base_r = reg_by_task_model_split[task][best_name].get(split)
            xgb_r = xgb_by_task.get(task, {}).get(split)
            if base_r is None or xgb_r is None:
                continue
            improved = xgb_r["rmse"] < base_r["rmse"]
            lines.append(
                f"| {split} | {best_name} | {_fmt(base_r['rmse'])} | {_fmt(xgb_r['rmse'])} | "
                f"{_fmt(xgb_r['r2'])} | {'Yes' if improved else 'No'} |\n"
            )

    lines.append("\n## Classification (15 tasks)\n\n")
    lines.append(
        "| Task | Split | Baseline-best | Baseline ROC-AUC | Baseline PR-AUC | "
        "XGBoost ROC-AUC | XGBoost PR-AUC | Improved (val PR-AUC)? |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    improved_count = 0
    not_improved_count = 0
    for task in CLASSIFICATION_TASKS:
        base = baseline_best_cls.get(task)
        xgb_splits = xgb_by_task.get(task, {})
        if base is None or not xgb_splits:
            continue
        best_name = base["model_name"]
        # Get the baseline records per split for the chosen best model.
        cls_by_model_split = {}
        for r in baseline_records:
            if r["task"] == task:
                cls_by_model_split.setdefault(r["model_name"], {})[r["split"]] = r

        val_improved = None
        for split in ["val", "test"]:
            base_r = cls_by_model_split.get(best_name, {}).get(split)
            xgb_r = xgb_splits.get(split)
            if base_r is None or xgb_r is None:
                continue
            if base_r["pr_auc"] is None or xgb_r["pr_auc"] is None:
                improved_str = "n/a"
            else:
                improved = xgb_r["pr_auc"] > base_r["pr_auc"]
                improved_str = "Yes" if improved else "No"
                if split == "val":
                    val_improved = improved
            lines.append(
                f"| {task} | {split} | {best_name} | {_fmt(base_r['roc_auc'])} | {_fmt(base_r['pr_auc'])} | "
                f"{_fmt(xgb_r['roc_auc'])} | {_fmt(xgb_r['pr_auc'])} | {improved_str} |\n"
            )
        if val_improved is True:
            improved_count += 1
        elif val_improved is False:
            not_improved_count += 1

    lines.append(
        f"\n**Summary**: on val PR-AUC, XGBoost with `scale_pos_weight` improved on the "
        f"baseline-best in {improved_count} of {improved_count + not_improved_count} classification tasks "
        f"with a defined comparison, and did not improve in {not_improved_count}.\n"
    )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index
    trained_at = datetime.now(timezone.utc).isoformat()

    baseline_records = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))

    all_records = []
    for task in REGRESSION_TASKS:
        print(f"Training regression task: {task}")
        all_records.extend(run_regression_task(task, df, descriptors[DESCRIPTOR_COLUMNS], trained_at))
    for task in CLASSIFICATION_TASKS:
        print(f"Training classification task: {task}")
        all_records.extend(run_classification_task(task, df, descriptors[DESCRIPTOR_COLUMNS], trained_at))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_records)} records to {RESULTS_PATH}")

    write_report(all_records, baseline_records)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
