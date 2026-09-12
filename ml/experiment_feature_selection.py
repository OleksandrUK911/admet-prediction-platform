"""Feature selection for the high-dimensional Morgan fingerprint features
used by the `logistic_regression_fingerprints` candidate (ml/baseline.py,
ml/model_registry.py) - per TODO/ml/TODO_feature_engineering.md's open item:
"Feature selection для high-dimensional toxicity fingerprints (variance
threshold, кореляційний фільтр)".

The 1024-bit Morgan fingerprint (ml/features.py's compute_fingerprints,
radius=2/1024 bits, the project default) is a sparse, high-dimensional,
binary feature space. Two classic, cheap filters are tried here, fit on
each task's TRAIN split only:

1. VarianceThreshold (sklearn.feature_selection) - drops bits that are
   (near-)constant across train molecules. For a binary feature,
   Var = p(1-p), so a threshold of 0.01 drops bits present in <~1.02% or
   >~98.98% of train molecules - these carry almost no discriminative
   signal for any classifier and are pure overhead.
2. A greedy pairwise-correlation filter applied on top of the
   variance-surviving bits: bits are visited in index order, and a bit is
   dropped the moment it is >0.95 (absolute Pearson r) correlated with any
   bit already kept. This is the standard "redundant feature" filter -
   Morgan fingerprint bits are not independent (overlapping circular
   substructures set several bits together), so some pairs are almost
   perfectly redundant.

Compares val PR-AUC (the same metric ml/model_registry.py's pick_winner()
uses for classification, since these tasks are imbalanced - see its
docstring) of Logistic Regression trained on the FULL 1024-bit fingerprint
vs. on the filtered subset, on 4 representative classification tasks:
the same 3 as ml/feature_engineering.py's fingerprint-size comparison
(NR-AR: small/imbalanced Tox21 assay, SR-MMP: larger/more-balanced Tox21
assay, bbbp_penetration: small non-Tox21 task) plus ct_tox (ClinTox,
small + heavily imbalanced, a 4th distinct profile).

This is exploratory only, in the same spirit as ml/feature_engineering.py
and ml/calibration.py: it does NOT modify ml/features.py or
ml/model_registry.py, and does NOT promote a feature-selected model into
production. An honest result either way (helps, hurts, or a wash) is the
expected, reportable outcome - not a foregone conclusion.

Usage:
    python ml/experiment_feature_selection.py

Writes ml/results/feature_selection_report.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import compute_fingerprints, task_rows

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
RESULTS_DIR = ROOT / "ml" / "results"
REPORT_PATH = RESULTS_DIR / "feature_selection_report.md"

SEED = 42
REPRESENTATIVE_TASKS = ["NR-AR", "SR-MMP", "bbbp_penetration", "ct_tox"]
N_BITS = 1024
VARIANCE_THRESHOLD = 0.01  # p(1-p) < 0.01 <=> bit "on" in <~1.02% or >~98.98% of train molecules
CORRELATION_THRESHOLD = 0.95
MEANINGFUL_DELTA = 0.02  # val PR-AUC gap below this is treated as noise (same bar as ml/feature_engineering.py)


def greedy_correlation_filter(X: np.ndarray, threshold: float = CORRELATION_THRESHOLD) -> np.ndarray:
    """Visits columns of X in order, keeping a column unless it is
    correlated above `threshold` (absolute Pearson r) with a column
    already kept. Returns a boolean keep-mask over X's columns.

    Constant columns (std==0, which VarianceThreshold should already have
    removed upstream, but guarded here too) are dropped outright - a
    correlation with a constant column is undefined (0/0)."""
    stds = X.std(axis=0)
    keep = np.zeros(X.shape[1], dtype=bool)
    kept_cols: list[np.ndarray] = []
    for j in range(X.shape[1]):
        if stds[j] == 0:
            continue
        col = X[:, j]
        is_redundant = False
        for kept_col in kept_cols:
            r = np.corrcoef(col, kept_col)[0, 1]
            if abs(r) > threshold:
                is_redundant = True
                break
        if not is_redundant:
            keep[j] = True
            kept_cols.append(col)
    return keep


def evaluate_task(df: pd.DataFrame, task: str) -> dict:
    rows = task_rows(df, task)
    y = rows[task].to_numpy().astype(int)
    fps = compute_fingerprints(rows["canonical_smiles"], radius=2, n_bits=N_BITS)
    train_mask = (rows["split"] == "train").to_numpy()
    val_mask = (rows["split"] == "val").to_numpy()

    X_train, y_train = fps[train_mask], y[train_mask]
    X_val, y_val = fps[val_mask], y[val_mask]

    # Baseline: all 1024 bits, no selection.
    lr_full = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, y_train)
    proba_full = lr_full.predict_proba(X_val)[:, 1]
    pr_auc_full = float(average_precision_score(y_val, proba_full))

    # Stage 1: VarianceThreshold, fit on train only.
    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD).fit(X_train)
    variance_keep = vt.get_support()
    n_after_variance = int(variance_keep.sum())

    # Stage 2: greedy correlation filter on the variance survivors, fit on train only.
    X_train_vt = X_train[:, variance_keep]
    correlation_keep_within_vt = greedy_correlation_filter(X_train_vt, CORRELATION_THRESHOLD)
    # Map back to the full 1024-bit index space.
    final_keep = np.zeros(N_BITS, dtype=bool)
    final_keep[np.where(variance_keep)[0]] = correlation_keep_within_vt
    n_final = int(final_keep.sum())

    X_train_sel = X_train[:, final_keep]
    X_val_sel = X_val[:, final_keep]
    lr_sel = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train_sel, y_train)
    proba_sel = lr_sel.predict_proba(X_val_sel)[:, 1]
    pr_auc_sel = float(average_precision_score(y_val, proba_sel))

    return {
        "task": task,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_bits_full": N_BITS,
        "n_bits_after_variance": n_after_variance,
        "n_bits_final": n_final,
        "pct_reduction": 100.0 * (1 - n_final / N_BITS),
        "pr_auc_full": pr_auc_full,
        "pr_auc_selected": pr_auc_sel,
        "delta": pr_auc_sel - pr_auc_full,
    }


def write_report(records: list[dict]) -> None:
    lines = ["# Feature selection report (high-dimensional fingerprints)\n\n"]
    lines.append(
        "Logistic Regression on Morgan fingerprints (radius=2, 1024 bits, "
        "ml/features.py's default), val PR-AUC, for 4 representative "
        "classification tasks (NR-AR: small/imbalanced Tox21 assay, SR-MMP: "
        "larger/more-balanced Tox21 assay, bbbp_penetration: small non-Tox21 "
        "task, ct_tox: small + heavily imbalanced ClinTox task). PR-AUC, not "
        "ROC-AUC, to match ml/model_registry.py's pick_winner() criterion "
        "for imbalanced classification tasks.\n\n"
        f"Pipeline: VarianceThreshold(threshold={VARIANCE_THRESHOLD}) fit on "
        "train only (drops bits with p(1-p) below that, i.e. near-constant "
        "across train molecules), then a greedy pairwise-correlation filter "
        f"(drop a bit the moment it is >{CORRELATION_THRESHOLD} absolute "
        "Pearson-r correlated with a bit already kept), also fit on train "
        f"only. 'Meaningful' is a val PR-AUC change of more than "
        f"{MEANINGFUL_DELTA} - smaller gaps are treated as noise given a "
        "single train/val split, same bar as ml/feature_engineering.py.\n\n"
    )
    lines.append(
        "| Task | N train | N val | Bits (full) | Bits (post variance) | "
        "Bits (final) | Reduction | PR-AUC (full) | PR-AUC (selected) | "
        "Delta | Meaningful? |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    verdicts = []
    for r in records:
        meaningful = abs(r["delta"]) > MEANINGFUL_DELTA
        verdicts.append((r["task"], r["delta"], meaningful))
        lines.append(
            f"| {r['task']} | {r['n_train']} | {r['n_val']} | {r['n_bits_full']} | "
            f"{r['n_bits_after_variance']} | {r['n_bits_final']} | "
            f"{r['pct_reduction']:.1f}% | {r['pr_auc_full']:.4f} | "
            f"{r['pr_auc_selected']:.4f} | {r['delta']:+.4f} | "
            f"{'YES' if meaningful else 'no (noise)'} |\n"
        )

    avg_reduction = float(np.mean([r["pct_reduction"] for r in records]))
    n_helped = sum(1 for _, d, m in verdicts if m and d > 0)
    n_hurt = sum(1 for _, d, m in verdicts if m and d < 0)
    n_wash = sum(1 for _, _, m in verdicts if not m)
    n_corr_filter_removed_nothing = sum(
        1 for r in records if r["n_bits_final"] == r["n_bits_after_variance"]
    )

    lines.append("\n## Verdict\n\n")
    lines.append(
        f"Average bit-count reduction across the 4 tasks: **{avg_reduction:.1f}%** "
        f"(from {N_BITS} bits down to the 'Bits (final)' column above per task). "
        f"Of the 4 tasks: {n_helped} meaningfully improved with feature selection, "
        f"{n_hurt} meaningfully got worse, {n_wash} were a wash (within "
        f"{MEANINGFUL_DELTA} val PR-AUC, i.e. noise given a single split).\n\n"
    )
    if n_corr_filter_removed_nothing == len(records):
        lines.append(
            f"Notable sub-finding: on all {len(records)} tasks, essentially all of "
            "the reduction came from VarianceThreshold alone - the correlation "
            f"filter (|r| > {CORRELATION_THRESHOLD}) removed zero additional bits "
            "beyond what VarianceThreshold already dropped. Morgan fingerprint "
            "bits, once the near-constant ones are removed, are apparently not "
            "redundant enough at this threshold/task-subset size to trigger the "
            "correlation filter - a real (if unglamorous) finding, not a bug: "
            "raising this threshold's usefulness would require either a lower "
            "|r| cutoff or a larger training set to estimate correlations more "
            "stably.\n\n"
        )
    elif n_corr_filter_removed_nothing > 0:
        lines.append(
            f"Sub-finding: on {n_corr_filter_removed_nothing}/{len(records)} tasks, "
            "the correlation filter removed zero additional bits beyond "
            "VarianceThreshold - most of the reduction came from the variance "
            "filter alone on this data.\n\n"
        )
    if n_helped > n_hurt and n_helped > 0:
        lines.append(
            "Feature selection reduced the bit count substantially without "
            "costing val PR-AUC on most tasks, and improved it on at least "
            "one - a plausible case for using it as a pre-processing step "
            "for the `logistic_regression_fingerprints` candidate, though "
            "**not** conclusive from a single train/val split (same caveat "
            "as ml/feature_engineering.py's fingerprint-size comparison). "
            "Not adopted as the production default here - this script is "
            "exploratory only and does not modify ml/features.py or "
            "ml/model_registry.py.\n"
        )
    elif n_hurt > n_helped and n_hurt > 0:
        lines.append(
            "Feature selection cost val PR-AUC on more tasks than it helped. "
            "The dropped bits, even the rare/constant/redundant-looking ones, "
            "carried some signal Logistic Regression's L2 regularization was "
            "already handling adequately on the full 1024-bit space - cutting "
            "them out did not pay for itself here. Recommendation: do not "
            "adopt this filter for the current logistic-regression-on-"
            "fingerprints candidate.\n"
        )
    else:
        lines.append(
            "A substantial bit-count reduction was achieved with essentially "
            "no change in val PR-AUC on any of the 4 tasks (all within the "
            f"{MEANINGFUL_DELTA} noise band) - a wash, not a clear win or "
            "loss. Feature selection here mostly reduces model size/inference "
            "cost, not accuracy, for `logistic_regression_fingerprints`. Not "
            "adopted as the production default, since the current pipeline "
            "does not have a size/latency problem that would justify the "
            "added complexity for zero accuracy gain.\n"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    records = []
    for task in REPRESENTATIVE_TASKS:
        print(f"Evaluating feature selection for task={task}...")
        record = evaluate_task(df, task)
        records.append(record)
        print(
            f"  bits: {record['n_bits_full']} -> {record['n_bits_final']} "
            f"({record['pct_reduction']:.1f}% reduction), "
            f"PR-AUC: {record['pr_auc_full']:.4f} -> {record['pr_auc_selected']:.4f} "
            f"(delta {record['delta']:+.4f})"
        )
    write_report(records)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
