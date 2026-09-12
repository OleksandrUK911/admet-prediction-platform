"""SMOTE / oversampling experiment for the most-imbalanced classification
tasks, per TODO/ml/TODO_experiments_per_task_models.md's "Експеримент з
SMOTE / oversampling для найбільш незбалансованих задач".

This is an EXPLORATORY, standalone experiment - like ml/feature_engineering.py
and ml/experiments_multitask.py, it does not touch ml/model_registry.py or
change what gets registered/served in production regardless of outcome; it
only writes a comparison report.

Scope: the 4 most-imbalanced classification tasks by actual train-split
positive rate (recomputed here, not assumed - see PICK_MOST_IMBALANCED
below): NR-PPAR-gamma, NR-AR-LBD, SR-ATAD5, NR-AR (2.9%-4.2% positive rate;
`ml/results/data_quality_report.md` and `ml/results/baseline_metrics.json`
were checked but positive-rate-per-task isn't tabulated verbatim there, so
prevalence is recomputed directly from data/processed/admet_processed.csv).

Method: Morgan fingerprints (same features `ml/features.py` computes for
every other script here). For each task, the SAME classifier architecture as
that task's currently-registered production winner is reused via
`ml/model_registry.py`'s `build_classifier` (so this experiment asks "does
SMOTE help THIS task's actual model type", not some generic classifier) -
fit once on the plain train split, once on a SMOTE-oversampled version of the
train split (minority class oversampled to parity via imbalanced-learn's
SMOTE). Both are then scored on the (untouched) val split by PR-AUC, the
metric used for model selection everywhere else in this project.

Requires imbalanced-learn - NOT in the main requirements.txt (see
ml/requirements-smote.txt, same "keep heavy/occasional deps separate"
pattern as ml/requirements-multitask.txt's torch):
    pip install -r ml/requirements-smote.txt

Usage:
    python ml/experiment_smote.py

Reads data/processed/admet_processed.csv.
Writes ml/results/smote_experiment_metrics.json and
ml/results/smote_experiment_report.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CLASSIFICATION_TASKS, compute_fingerprints, task_rows
from model_registry import build_classifier, evaluate_classification

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "imbalanced-learn is required for this experiment and is deliberately kept out of "
        "the main requirements.txt (heavy/occasional dependency, same pattern as torch in "
        "ml/requirements-multitask.txt). Install it with:\n"
        "    pip install -r ml/requirements-smote.txt"
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
PRODUCTION_METADATA = ROOT / "models" / "production" / "metadata.json"
RESULTS_DIR = ROOT / "ml" / "results"
METRICS_PATH = RESULTS_DIR / "smote_experiment_metrics.json"
REPORT_PATH = RESULTS_DIR / "smote_experiment_report.md"
SEED = 42
N_MOST_IMBALANCED = 4
SMOTE_K_NEIGHBORS = 5  # imbalanced-learn's default - needs >= k+1 minority samples in train


def most_imbalanced_tasks(df: pd.DataFrame, n: int) -> list[tuple[str, float, int]]:
    """Actual train-split positive rate per classification task, ascending
    (most imbalanced first) - computed directly rather than assumed from
    memory of earlier reports."""
    rates = []
    for task in CLASSIFICATION_TASKS:
        rows = task_rows(df, task)
        train_rows = rows[rows["split"] == "train"]
        y = train_rows[task].to_numpy().astype(int)
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        rates.append((task, float(y.mean()), int(y.sum())))
    rates.sort(key=lambda r: r[1])
    return rates[:n]


def production_model_name(task: str, metadata: dict) -> str | None:
    winners = metadata.get("per_task_winners", {})
    info = winners.get(task)
    return info["model_name"] if info else None


def run_task(task: str, model_name: str, df: pd.DataFrame, fingerprints: np.ndarray) -> dict:
    rows = task_rows(df, task)
    row_positions = df.index.get_indexer(rows.index)
    X = fingerprints[row_positions]
    y = rows[task].to_numpy().astype(int)

    train_mask = (rows["split"] == "train").to_numpy()
    val_mask = (rows["split"] == "val").to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    n_pos_train = int(y_train.sum())
    n_neg_train = int(len(y_train) - n_pos_train)

    # Without SMOTE - the same construction ml/model_registry.py would use
    # for this task's model_name (class-weighting for xgboost, none for the
    # others - matching production, not re-litigating that choice here).
    baseline_model = build_classifier(model_name, y_train).fit(X_train, y_train)
    baseline_metrics = evaluate_classification(baseline_model, X_val, y_val)

    result = {
        "task": task,
        "model_name": model_name,
        "n_train": len(y_train),
        "n_pos_train": n_pos_train,
        "n_neg_train": n_neg_train,
        "train_positive_rate": n_pos_train / len(y_train),
        "baseline_val": baseline_metrics,
    }

    if n_pos_train <= SMOTE_K_NEIGHBORS:
        result["smote_val"] = None
        result["smote_skipped_reason"] = (
            f"only {n_pos_train} positive train examples <= SMOTE's k_neighbors={SMOTE_K_NEIGHBORS} - "
            "cannot synthesize neighbors for the minority class"
        )
        result["smote_pr_auc_delta"] = None
        return result

    smote = SMOTE(random_state=SEED, k_neighbors=SMOTE_K_NEIGHBORS)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    smote_model = build_classifier(model_name, y_resampled).fit(X_resampled, y_resampled)
    smote_metrics = evaluate_classification(smote_model, X_val, y_val)

    result["n_train_after_smote"] = len(y_resampled)
    result["smote_val"] = smote_metrics
    result["smote_skipped_reason"] = None
    base_pr = baseline_metrics.get("pr_auc")
    smote_pr = smote_metrics.get("pr_auc")
    result["smote_pr_auc_delta"] = (smote_pr - base_pr) if (base_pr is not None and smote_pr is not None) else None
    return result


def write_report(results: list[dict]) -> None:
    lines = [
        "# SMOTE Oversampling Experiment (Most-Imbalanced Classification Tasks)",
        "",
        (
            f"The {N_MOST_IMBALANCED} classification tasks with the lowest actual train-split "
            "positive rate (recomputed directly from `data/processed/admet_processed.csv`, not "
            "assumed) get a SMOTE-oversampled training run, using the SAME classifier "
            "architecture as that task's current production winner "
            "(`models/production/metadata.json` -> `ml/model_registry.py`'s `build_classifier`), "
            "on Morgan fingerprint features. Both the plain-train and SMOTE-oversampled model are "
            "scored on the untouched val split by PR-AUC - the metric this project uses for model "
            "selection throughout."
        ),
        "",
        (
            "This is exploratory only - it does NOT change what is registered in "
            "`models/production/models.joblib`, regardless of outcome."
        ),
        "",
        "## Results",
        "",
        "| task | model | train n (pos/neg) | pos rate | val PR-AUC (no SMOTE) | val PR-AUC (SMOTE) | delta | helped? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    n_helped, n_hurt, n_skipped = 0, 0, 0
    for r in results:
        base_pr = r["baseline_val"].get("pr_auc")
        base_str = f"{base_pr:.3f}" if base_pr is not None else "n/a"
        if r["smote_val"] is None:
            n_skipped += 1
            lines.append(
                f"| {r['task']} | {r['model_name']} | {r['n_pos_train']}/{r['n_neg_train']} | "
                f"{r['train_positive_rate']:.1%} | {base_str} | skipped | - | skipped |"
            )
            continue
        smote_pr = r["smote_val"].get("pr_auc")
        smote_str = f"{smote_pr:.3f}" if smote_pr is not None else "n/a"
        delta = r["smote_pr_auc_delta"]
        delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
        if delta is not None and delta > 0.005:
            verdict = "Yes"
            n_helped += 1
        elif delta is not None and delta < -0.005:
            verdict = "No (worse)"
            n_hurt += 1
        else:
            verdict = "~no change"
        lines.append(
            f"| {r['task']} | {r['model_name']} | {r['n_pos_train']}/{r['n_neg_train']} | "
            f"{r['train_positive_rate']:.1%} | {base_str} | {smote_str} | {delta_str} | {verdict} |"
        )

    lines += ["", "## Honest result", ""]
    n_scored = len(results) - n_skipped
    lines.append(
        f"Of {n_scored} task(s) where SMOTE could run, it improved val PR-AUC (by more than a "
        f"0.005 margin) on {n_helped}, made it measurably worse on {n_hurt}, and left it "
        f"essentially unchanged on {n_scored - n_helped - n_hurt}. {n_skipped} task(s) were "
        "skipped (too few minority-class training examples for SMOTE's neighbor search)."
    )
    lines.append("")
    if n_helped == 0:
        lines.append(
            "**SMOTE did not help on any of these tasks.** This matches the pattern already "
            "documented in `ml/TODO_experiments_per_task_models.md` for class-weighting "
            "(\"лише 3/15 задач покращились\") and this project's general finding that these "
            "tasks' classifiers (logistic regression / random forest / XGBoost, already "
            "class-weighted where relevant) are not starved for minority-class signal so much as "
            "for genuinely informative features on the rarest assays - synthesizing more minority "
            "examples in fingerprint bit-space does not add real chemical information, and for "
            "tree/linear models already handling the imbalance reasonably (via `scale_pos_weight` "
            "or, for logistic regression/random forest here, implicitly via probability "
            "calibration downstream in `ml/model_registry.py`) there is little headroom left for "
            "oversampling to capture. This is reported plainly rather than forced into a positive "
            "narrative, matching how multitask-vs-per-task and isotonic-vs-Platt were reported in "
            "this project."
        )
    elif n_helped < n_scored:
        lines.append(
            f"**Mixed result**: SMOTE helped on {n_helped}/{n_scored} scored tasks and did not on "
            "the rest - reported as-is rather than generalized into a blanket recommendation "
            "either way."
        )
    else:
        lines.append(
            f"SMOTE improved val PR-AUC on all {n_scored} scored tasks - reported as-is, though "
            "with only 4 tasks and one val split each, treat this as a directional signal, not a "
            "guarantee it generalizes to a held-out re-split or to other tasks."
        )

    lines += [
        "",
        "## Caveats",
        "",
        (
            "- SMOTE is applied on 1024-bit Morgan fingerprint features - synthetic neighbors are "
            "interpolated bit vectors that do not correspond to real molecules; this is standard "
            "practice for SMOTE but worth stating plainly since \"chemical validity\" of the "
            "synthesized examples is not checked."
        ),
        (
            "- Val split and model architecture are identical to production - only the training-set "
            "class balance changes, so any effect observed is attributable to oversampling, not a "
            "confound from a different model or split."
        ),
        "- This experiment is not wired into ml/model_registry.py and does not change what ships.",
        (
            "- Both rows (no-SMOTE and SMOTE) here use fingerprint features for every task, "
            "including tasks whose actual production winner is `random_forest_descriptors` (i.e. "
            "normally trained on the 7 physico-chemical descriptors, not fingerprints) - this is "
            "deliberate (SMOTE needs one consistent feature space to compare fairly), but it means "
            "the \"no-SMOTE\" PR-AUC in this report can differ from that task's real "
            "`models/production/metadata.json` val PR-AUC, which may use descriptors instead. The "
            "comparison that matters here is no-SMOTE vs SMOTE, both on the same fingerprint "
            "features - not this report's no-SMOTE column vs production metadata."
        ),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    fingerprints = compute_fingerprints(df["canonical_smiles"])
    metadata = json.loads(PRODUCTION_METADATA.read_text(encoding="utf-8"))

    picked = most_imbalanced_tasks(df, N_MOST_IMBALANCED)
    print(f"Most imbalanced {N_MOST_IMBALANCED} tasks by actual train positive rate:")
    for task, rate, n_pos in picked:
        print(f"  {task}: {rate:.1%} positive ({n_pos} positives in train)")

    results = []
    for task, _, _ in picked:
        model_name = production_model_name(task, metadata)
        if model_name is None:
            print(f"  Skipping {task}: no registered production winner found in metadata.json")
            continue
        result = run_task(task, model_name, df, fingerprints)
        results.append(result)
        base = result["baseline_val"].get("pr_auc")
        smote = result["smote_val"]["pr_auc"] if result["smote_val"] else None
        print(f"  {task} ({model_name}): val PR-AUC no-SMOTE={base:.3f}" + (f" SMOTE={smote:.3f}" if smote is not None else " SMOTE=skipped"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    print(f"\nWrote {METRICS_PATH}\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
