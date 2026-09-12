"""Rigorous, whole-portfolio evaluation of the REGISTERED production models
(models/production/models.joblib + metadata.json), across all 16 tasks.

Complements ml/results/per_task_comparison_report.md (which only compares
baseline vs XGBoost *candidates* pre-registry) with a pass over the actual
final, calibrated, registered winner per task:

  1. Load the real bundle for each task and evaluate it on train/val/test -
     a consistency check against models/production/metadata.json's stored
     val/test numbers (which the registry already computed once), plus a
     NEW train-vs-val gap check (overfitting risk) that the registry never
     reports.
  2. 5-fold CV on the train+val pool (test never touched) for 3 representative
     tasks - solubility, SR-MMP, bbbp_penetration - reusing each task's
     winning model construction from ml/model_registry.py's build_classifier /
     retrain_regressor, to see whether the single train/val/test split's
     metric is a stable estimate or a lucky/unlucky draw.
  3. A whole-portfolio ranking (classification tasks by test PR-AUC, the one
     regression task by test R2) plus a one-line diagnosis for the 3 worst
     classification tasks.

Usage:
    python ml/evaluate.py

Reads models/production/{models.joblib,metadata.json} + data/processed/admet_processed.csv.
Writes ml/results/full_evaluation_report.md and ml/results/cross_validation_metrics.json.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    REGRESSION_TASKS,
    compute_descriptors,
    compute_fingerprints,
    task_rows,
)
from model_registry import (
    build_classifier,
    evaluate_classification,
    evaluate_regression,
    feature_type_for,
    retrain_regressor,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
PRODUCTION_DIR = ROOT / "models" / "production"
RESULTS_DIR = ROOT / "ml" / "results"
REPORT_PATH = RESULTS_DIR / "full_evaluation_report.md"
CV_METRICS_PATH = RESULTS_DIR / "cross_validation_metrics.json"
SEED = 42
N_FOLDS = 5
# Overfitting flag thresholds - "dramatically better" train performance than val.
CLASSIFICATION_GAP_THRESHOLD = 0.15  # train PR-AUC - val PR-AUC
REGRESSION_GAP_RATIO_THRESHOLD = 0.5  # (val RMSE - train RMSE) / val RMSE

# The 3 representative tasks for part 2's CV stability check (one regression,
# one large/balanced Tox21 classification task, one smaller non-Tox21 task) -
# picked per the task brief, not re-derived here.
CV_TASKS = ["solubility", "SR-MMP", "bbbp_penetration"]


def close_enough(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(b))


def evaluate_all_tasks(df: pd.DataFrame, descriptors: pd.DataFrame, fingerprints: np.ndarray, bundle: dict) -> dict:
    """Reload each task's registered model bundle and compute train/val/test
    metrics with it directly (not a freshly retrained copy)."""
    task_bundles = bundle["tasks"]
    per_task = {}

    for task in REGRESSION_TASKS + CLASSIFICATION_TASKS:
        if task not in task_bundles:
            continue
        info = task_bundles[task]
        model_name = info["model_name"]
        is_classification = info["task_type"] == "classification"

        rows = task_rows(df, task)
        row_positions = df.index.get_indexer(rows.index)
        desc = descriptors.loc[rows.index, DESCRIPTOR_COLUMNS].to_numpy()
        fps = fingerprints[row_positions]
        X = fps if feature_type_for(model_name) == "fingerprints" else desc

        train_mask = (rows["split"] == "train").to_numpy()
        val_mask = (rows["split"] == "val").to_numpy()
        test_mask = (rows["split"] == "test").to_numpy()

        if is_classification:
            y = rows[task].to_numpy().astype(int)
            model = info["model"]
            train_metrics = evaluate_classification(model, X[train_mask], y[train_mask])
            val_metrics = evaluate_classification(model, X[val_mask], y[val_mask])
            test_metrics = evaluate_classification(model, X[test_mask], y[test_mask]) if test_mask.sum() else {"roc_auc": None, "pr_auc": None}
        else:
            y = rows[task].to_numpy()
            model, scaler = info["model"], info.get("scaler")
            train_metrics = evaluate_regression(model, X[train_mask], y[train_mask], scaler)
            val_metrics = evaluate_regression(model, X[val_mask], y[val_mask], scaler)
            test_metrics = evaluate_regression(model, X[test_mask], y[test_mask], scaler) if test_mask.sum() else {}

        n_positive_total = int(y.sum()) if is_classification else None
        n_positive_train = int(y[train_mask].sum()) if is_classification else None

        per_task[task] = {
            "model_name": model_name,
            "task_type": info["task_type"],
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_positive_total": n_positive_total,
            "n_positive_train": n_positive_train,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
    return per_task


def check_consistency(per_task: dict, metadata: dict) -> list[dict]:
    """Compare our freshly-computed val/test metrics against the numbers
    ml/model_registry.py already stored in metadata.json - should match
    (or be very close - small floating point drift is expected, but a large
    mismatch means a bug in how the bundle is being reloaded/evaluated here)."""
    mismatches = []
    for task, recorded in metadata["per_task_winners"].items():
        ours = per_task.get(task)
        if ours is None:
            mismatches.append({"task": task, "issue": "missing from reload"})
            continue
        for split in ("val", "test"):
            for metric, recorded_value in recorded[split].items():
                our_value = ours[split].get(metric)
                if not close_enough(our_value, recorded_value, tol=1e-4):
                    mismatches.append(
                        {
                            "task": task,
                            "split": split,
                            "metric": metric,
                            "recorded": recorded_value,
                            "reloaded": our_value,
                        }
                    )
    return mismatches


def overfitting_flags(per_task: dict) -> list[dict]:
    flags = []
    for task, info in per_task.items():
        if info["task_type"] == "classification":
            train_pr = info["train"].get("pr_auc")
            val_pr = info["val"].get("pr_auc")
            if train_pr is None or val_pr is None:
                continue
            gap = train_pr - val_pr
            if gap >= CLASSIFICATION_GAP_THRESHOLD:
                flags.append(
                    {
                        "task": task,
                        "metric": "pr_auc",
                        "train": train_pr,
                        "val": val_pr,
                        "gap": gap,
                    }
                )
        else:
            train_rmse = info["train"].get("rmse")
            val_rmse = info["val"].get("rmse")
            if train_rmse is None or val_rmse is None or val_rmse == 0:
                continue
            gap_ratio = (val_rmse - train_rmse) / val_rmse
            if gap_ratio >= REGRESSION_GAP_RATIO_THRESHOLD:
                flags.append(
                    {
                        "task": task,
                        "metric": "rmse",
                        "train": train_rmse,
                        "val": val_rmse,
                        "gap_ratio": gap_ratio,
                    }
                )
    return flags


def run_cv(df: pd.DataFrame, descriptors: pd.DataFrame, fingerprints: np.ndarray, per_task: dict) -> list[dict]:
    """5-fold CV on train+val (test untouched) for the 3 representative tasks,
    reusing each task's winning model construction from ml/model_registry.py."""
    records = []
    for task in CV_TASKS:
        model_name = per_task[task]["model_name"]
        is_classification = task in CLASSIFICATION_TASKS

        rows = task_rows(df, task)
        row_positions = df.index.get_indexer(rows.index)
        pool_mask = rows["split"].isin(["train", "val"]).to_numpy()
        desc = descriptors.loc[rows.index, DESCRIPTOR_COLUMNS].to_numpy()
        fps = fingerprints[row_positions]
        X_full = fps if feature_type_for(model_name) == "fingerprints" else desc
        X = X_full[pool_mask]
        y = rows[task].to_numpy()[pool_mask]
        if is_classification:
            y = y.astype(int)

        if is_classification:
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            split_iter = splitter.split(X)

        for fold, (train_idx, test_idx) in enumerate(split_iter):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if is_classification:
                model = build_classifier(model_name, y_train).fit(X_train, y_train)
                metrics = evaluate_classification(model, X_test, y_test)
                for metric_name, value in metrics.items():
                    records.append({"task": task, "fold": fold, "metric_name": metric_name, "value": value})
            else:
                model, scaler = retrain_regressor(model_name, X_train, y_train)
                metrics = evaluate_regression(model, X_test, y_test, scaler)
                for metric_name, value in metrics.items():
                    records.append({"task": task, "fold": fold, "metric_name": metric_name, "value": value})
    return records


def summarize_cv(cv_records: list[dict]) -> dict:
    summary = {}
    for task in CV_TASKS:
        summary[task] = {}
        metric_names = sorted({r["metric_name"] for r in cv_records if r["task"] == task})
        for metric_name in metric_names:
            values = [r["value"] for r in cv_records if r["task"] == task and r["metric_name"] == metric_name]
            summary[task][metric_name] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n_folds": len(values)}
    return summary


def diagnose_worst_classification_tasks(per_task: dict, n: int = 3) -> list[dict]:
    classification = [(task, info) for task, info in per_task.items() if info["task_type"] == "classification"]
    ranked = sorted(classification, key=lambda kv: (kv[1]["test"].get("pr_auc") if kv[1]["test"].get("pr_auc") is not None else 1.0))
    worst = ranked[:n]
    diagnoses = []
    for task, info in worst:
        n_pos = info["n_positive_total"]
        n_total = info["n_train"] + info["n_val"] + info["n_test"]
        prevalence = n_pos / n_total if n_total else 0.0
        if prevalence < 0.05:
            reason = f"very few positives ({n_pos}/{n_total} = {prevalence:.1%} prevalence) - PR-AUC is hard to beat the low base rate with this little signal"
        elif n_total < 1500:
            reason = f"small dataset ({n_total} labeled rows total) limits how much the model can learn"
        else:
            reason = f"test PR-AUC ({info['test'].get('pr_auc'):.3f}) close to the task's positive prevalence ({prevalence:.1%}) - near-baseline performance"
        diagnoses.append(
            {
                "task": task,
                "test_pr_auc": info["test"].get("pr_auc"),
                "n_positive": n_pos,
                "n_total": n_total,
                "prevalence": prevalence,
                "diagnosis": reason,
            }
        )
    return diagnoses


def write_report(per_task: dict, mismatches: list[dict], flags: list[dict], cv_summary: dict, worst: list[dict]) -> None:
    lines = ["# Full Production-Model Evaluation Report", ""]

    lines += [
        "## 1. Full production-model evaluation (train/val/test, reloaded bundles)",
        "",
        (
            "Metrics recomputed by loading the actual registered bundle from "
            "`models/production/models.joblib` for each of the 16 tasks (not a "
            "freshly-retrained copy) and scoring it on train/val/test."
        ),
        "",
        "| task | model | n_train | n_val | n_test | train metric | val metric | test metric |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task in REGRESSION_TASKS + CLASSIFICATION_TASKS:
        if task not in per_task:
            continue
        info = per_task[task]
        if info["task_type"] == "classification":
            fmt = lambda m: f"PR-AUC={m.get('pr_auc'):.3f}" if m.get("pr_auc") is not None else "n/a"
        else:
            fmt = lambda m: f"RMSE={m.get('rmse'):.3f}" if m.get("rmse") is not None else "n/a"
        lines.append(
            f"| {task} | {info['model_name']} | {info['n_train']} | {info['n_val']} | {info['n_test']} "
            f"| {fmt(info['train'])} | {fmt(info['val'])} | {fmt(info['test'])} |"
        )

    lines += ["", "### Consistency check vs `models/production/metadata.json`", ""]
    if not mismatches:
        lines.append("All reloaded val/test metrics match the stored `per_task_winners` metadata (within 1e-4 relative tolerance).")
    else:
        lines.append(f"{len(mismatches)} mismatch(es) found between reloaded metrics and stored metadata:")
        lines.append("")
        for m in mismatches:
            lines.append(f"- {m}")

    lines += ["", "### Train-vs-val overfitting gap check", ""]
    if not flags:
        lines.append(
            f"No task shows a train-vs-val gap above threshold "
            f"(classification: train PR-AUC - val PR-AUC >= {CLASSIFICATION_GAP_THRESHOLD}; "
            f"regression: (val RMSE - train RMSE) / val RMSE >= {REGRESSION_GAP_RATIO_THRESHOLD})."
        )
    else:
        lines.append("Flagged tasks (train performance dramatically better than val - overfitting risk):")
        lines.append("")
        for f in flags:
            if f["metric"] == "pr_auc":
                lines.append(f"- **{f['task']}**: train PR-AUC={f['train']:.3f} vs val PR-AUC={f['val']:.3f} (gap={f['gap']:.3f})")
            else:
                lines.append(f"- **{f['task']}**: train RMSE={f['train']:.3f} vs val RMSE={f['val']:.3f} (ratio={f['gap_ratio']:.2f})")

    lines += [
        "",
        "## 2. Cross-validation stability check (5-fold, train+val pool only)",
        "",
        (
            "Test split never touched. Each task's winning model is reconstructed via "
            "`ml/model_registry.py`'s `build_classifier`/`retrain_regressor` (same "
            "architecture and hyperparameters as production, but NOT wrapped in "
            "`CalibratedClassifierCV` - that step only affects probability calibration, "
            "not ranking metrics like ROC-AUC/PR-AUC, so it is skipped here to isolate "
            "the base model's stability)."
        ),
        "",
        "| task | metric | mean | std | n_folds |",
        "|---|---|---|---|---|",
    ]
    for task in CV_TASKS:
        for metric_name, stats in cv_summary[task].items():
            lines.append(f"| {task} | {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['n_folds']} |")

    lines += [
        "",
        (
            "Interpretation: a std that is small relative to the gap between candidate "
            "models (or relative to the single val-split number quoted in "
            "`models/production/metadata.json`) suggests the reported val metric is a "
            "stable estimate; a std comparable to or larger than that gap suggests the "
            "single train/val/test split could have been a lucky or unlucky draw for "
            "that task."
        ),
        "",
    ]

    lines += ["## 3. Whole-portfolio ranking", "", "### Classification tasks, ranked by test PR-AUC (descending)", ""]
    lines += ["| rank | task | test PR-AUC | test ROC-AUC | n_positive/n_total |", "|---|---|---|---|---|"]
    classification = [(t, i) for t, i in per_task.items() if i["task_type"] == "classification"]
    ranked = sorted(classification, key=lambda kv: (kv[1]["test"].get("pr_auc") if kv[1]["test"].get("pr_auc") is not None else -1), reverse=True)
    for rank, (task, info) in enumerate(ranked, start=1):
        pr = info["test"].get("pr_auc")
        roc = info["test"].get("roc_auc")
        pr_str = f"{pr:.3f}" if pr is not None else "n/a"
        roc_str = f"{roc:.3f}" if roc is not None else "n/a"
        n_total = info["n_train"] + info["n_val"] + info["n_test"]
        lines.append(f"| {rank} | {task} | {pr_str} | {roc_str} | {info['n_positive_total']}/{n_total} |")

    lines += ["", "### Regression task, by test R2", ""]
    for task, info in per_task.items():
        if info["task_type"] != "regression":
            continue
        r2 = info["test"].get("r2")
        lines.append(f"- **{task}**: test R2 = {r2:.3f}, test RMSE = {info['test'].get('rmse'):.3f}")

    lines += ["", "### 3 worst classification tasks - diagnosis", ""]
    for w in worst:
        lines.append(
            f"- **{w['task']}** (test PR-AUC={w['test_pr_auc']:.3f}, "
            f"n_positive/n_total={w['n_positive']}/{w['n_total']} = {w['prevalence']:.1%}): {w['diagnosis']}"
        )

    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index
    fingerprints = compute_fingerprints(df["canonical_smiles"])

    bundle = joblib.load(PRODUCTION_DIR / "models.joblib")
    metadata = json.loads((PRODUCTION_DIR / "metadata.json").read_text(encoding="utf-8"))

    per_task = evaluate_all_tasks(df, descriptors, fingerprints, bundle)
    mismatches = check_consistency(per_task, metadata)
    flags = overfitting_flags(per_task)

    cv_records = run_cv(df, descriptors, fingerprints, per_task)
    CV_METRICS_PATH.write_text(json.dumps(cv_records, indent=2), encoding="utf-8")
    cv_summary = summarize_cv(cv_records)

    worst = diagnose_worst_classification_tasks(per_task)

    write_report(per_task, mismatches, flags, cv_summary, worst)

    print(f"Consistency check: {len(mismatches)} mismatch(es).")
    print(f"Overfitting flags: {len(flags)} task(s).")
    for task in CV_TASKS:
        for metric_name, stats in cv_summary[task].items():
            print(f"  CV {task} {metric_name}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print("3 worst classification tasks:")
    for w in worst:
        print(f"  {w['task']}: test PR-AUC={w['test_pr_auc']:.3f} - {w['diagnosis']}")
    print(f"\nWrote {REPORT_PATH}\nWrote {CV_METRICS_PATH}")


if __name__ == "__main__":
    main()
