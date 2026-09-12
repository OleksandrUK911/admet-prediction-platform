"""Probability calibration + applicability domain, for a representative
subset of the classification tasks (per ml/TODO_calibration.md).

This project's stated differentiator is CALIBRATED uncertainty, not just raw
predictions - a tree ensemble's predict_proba() output is known to not match
true label frequencies well, and calibration (Platt scaling / isotonic
regression) is the standard fix. This script demonstrates and measures that
fix on 4 tasks (not all 15 - that would be excessive for a first pass):
  - NR-AR: small, imbalanced Tox21 assay
  - SR-MMP: larger, more balanced Tox21 assay
  - bbbp_penetration: non-Tox21, moderate size
  - ct_tox: small ClinTox task

For each task it trains a Random Forest on the 7 shared descriptors (same
architecture as ml/baseline.py's RF baseline, for a fair comparison), wraps
it with sklearn's CalibratedClassifierCV (Platt/sigmoid and isotonic,
internal 5-fold cross-fitting on the TRAIN split only - val/test are never
touched during calibration, which would leak), and reports:
  1. Expected Calibration Error (ECE) and ROC-AUC on val, for raw RF vs.
     Platt-calibrated vs. isotonic-calibrated.
  2. An applicability-domain check: for test-split molecules, distance
     (in scaled descriptor space) to the nearest TRAIN molecule, and whether
     the most out-of-domain 10% of test molecules get worse predictions.

Usage:
    python ml/calibration.py

Writes ml/results/calibration_metrics.json and ml/results/calibration_report.md.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import DESCRIPTOR_COLUMNS, compute_descriptors, task_rows

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
RESULTS_PATH = ROOT / "ml" / "results" / "calibration_metrics.json"
REPORT_PATH = ROOT / "ml" / "results" / "calibration_report.md"
SEED = 42
TASKS = ["NR-AR", "SR-MMP", "bbbp_penetration", "ct_tox"]
N_BINS = 10
OOD_QUANTILE = 0.90  # top 10% nearest-neighbor distance = "out of domain"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """Bin predicted probabilities into n_bins equal-width bins, compare mean
    predicted probability to actual positive rate per bin, weighted by bin
    size. Standard ECE definition."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # rightmost edge inclusive
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def make_rf() -> RandomForestClassifier:
    # Same architecture as ml/baseline.py's RF baseline, for a fair comparison.
    return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)


def run_task(task: str, df: pd.DataFrame, descriptors: pd.DataFrame, trained_at: str) -> dict:
    rows = task_rows(df, task)
    desc = descriptors.loc[rows.index][DESCRIPTOR_COLUMNS].to_numpy()
    y = rows[task].to_numpy().astype(int)
    split = rows["split"].to_numpy()

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    X_train, y_train = desc[train_mask], y[train_mask]
    X_val, y_val = desc[val_mask], y[val_mask]
    X_test, y_test = desc[test_mask], y[test_mask]

    # --- Calibration comparison (train + val only) ---
    raw_rf = make_rf().fit(X_train, y_train)
    # cv="prefit" would require holding out a second slice of train purely
    # for calibration, shrinking an already-small fit set further for tasks
    # like ct_tox. Internal cross-fitting (cv=5) calibrates on out-of-fold
    # train predictions instead - still never touches val/test, and makes
    # better use of the limited train data.
    platt = CalibratedClassifierCV(make_rf(), method="sigmoid", cv=5).fit(X_train, y_train)
    isotonic = CalibratedClassifierCV(make_rf(), method="isotonic", cv=5).fit(X_train, y_train)

    calibration_rows = []
    for name, model in [("raw_rf", raw_rf), ("platt_sigmoid", platt), ("isotonic", isotonic)]:
        proba = model.predict_proba(X_val)[:, 1]
        calibration_rows.append({
            "task": task,
            "method": name,
            "n_val": int(val_mask.sum()),
            "ece": expected_calibration_error(y_val, proba),
            "roc_auc": float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else None,
        })

    # --- Applicability domain (test split, raw RF predictions) ---
    scaler = StandardScaler().fit(X_train)
    nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(X_train))
    nn_dist, _ = nn.kneighbors(scaler.transform(X_test))
    nn_dist = nn_dist[:, 0]

    test_proba = raw_rf.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    correct = (test_pred == y_test).astype(float)

    threshold = float(np.quantile(nn_dist, OOD_QUANTILE))
    ood_mask = nn_dist >= threshold
    in_domain_mask = ~ood_mask

    ad_result = {
        "task": task,
        "n_test": int(test_mask.sum()),
        "ood_threshold_distance": threshold,
        "n_ood": int(ood_mask.sum()),
        "ood_accuracy": float(correct[ood_mask].mean()) if ood_mask.sum() > 0 else None,
        "in_domain_accuracy": float(correct[in_domain_mask].mean()) if in_domain_mask.sum() > 0 else None,
        "ood_mean_nn_distance": float(nn_dist[ood_mask].mean()) if ood_mask.sum() > 0 else None,
        "in_domain_mean_nn_distance": float(nn_dist[in_domain_mask].mean()) if in_domain_mask.sum() > 0 else None,
    }

    return {"calibration": calibration_rows, "applicability_domain": ad_result, "trained_at": trained_at}


def write_report(results: list[dict]) -> None:
    lines = ["# Calibration + applicability domain report\n\n"]

    lines.append(
        "## Expected Calibration Error (ECE) comparison\n\n"
        "ECE measured on the val split (never used to fit or calibrate the "
        "model). Lower ECE = predicted probabilities better match true "
        "positive rates. ROC-AUC is included as a sanity check: calibration "
        "reshapes probabilities but should not change ranking, so ROC-AUC "
        "should stay roughly flat across the three methods.\n\n"
        "| Task | Method | N (val) | ECE | ROC-AUC |\n|---|---|---|---|---|\n"
    )
    for r in results:
        for c in r["calibration"]:
            auc = f"{c['roc_auc']:.3f}" if c["roc_auc"] is not None else "n/a"
            lines.append(f"| {c['task']} | {c['method']} | {c['n_val']} | {c['ece']:.3f} | {auc} |\n")

    lines.append("\n### Which method won, per task\n\n")
    for r in results:
        by_method = {c["method"]: c["ece"] for c in r["calibration"]}
        raw = by_method["raw_rf"]
        best_method = min(by_method, key=by_method.get)
        if best_method == "raw_rf":
            verdict = f"neither calibration method beat raw RF (raw ECE={raw:.3f} was lowest)"
        else:
            verdict = (
                f"{best_method} won (ECE={by_method[best_method]:.3f} vs. raw RF ECE={raw:.3f})"
            )
        lines.append(f"- **{r['calibration'][0]['task']}**: {verdict}\n")

    lines.append(
        "\n## Applicability domain (descriptor-space nearest-neighbor distance)\n\n"
        f"For each task's test-split molecules: Euclidean distance (in "
        f"train-fit-scaled 7-descriptor space) to the nearest TRAIN molecule "
        f"for that task. The top {int((1 - OOD_QUANTILE) * 100)}% by distance "
        "are flagged out-of-domain (OOD); raw-RF classification accuracy "
        "(threshold 0.5) is compared between the OOD group and the rest.\n\n"
        "| Task | N test | N OOD | OOD mean NN dist | In-domain mean NN dist | "
        "OOD accuracy | In-domain accuracy |\n|---|---|---|---|---|---|---|\n"
    )
    for r in results:
        ad = r["applicability_domain"]
        ood_acc = f"{ad['ood_accuracy']:.3f}" if ad["ood_accuracy"] is not None else "n/a"
        id_acc = f"{ad['in_domain_accuracy']:.3f}" if ad["in_domain_accuracy"] is not None else "n/a"
        lines.append(
            f"| {ad['task']} | {ad['n_test']} | {ad['n_ood']} | "
            f"{ad['ood_mean_nn_distance']:.3f} | {ad['in_domain_mean_nn_distance']:.3f} | "
            f"{ood_acc} | {id_acc} |\n"
        )

    lines.append("\n### Did applicability domain predict worse accuracy?\n\n")
    for r in results:
        ad = r["applicability_domain"]
        if ad["ood_accuracy"] is None or ad["in_domain_accuracy"] is None:
            lines.append(f"- **{ad['task']}**: not enough data in one of the groups to compare.\n")
            continue
        gap = ad["in_domain_accuracy"] - ad["ood_accuracy"]
        if gap > 0.02:
            lines.append(
                f"- **{ad['task']}**: yes - OOD accuracy {ad['ood_accuracy']:.3f} vs. "
                f"in-domain {ad['in_domain_accuracy']:.3f} (gap {gap:.3f}).\n"
            )
        elif gap < -0.02:
            lines.append(
                f"- **{ad['task']}**: no - OOD accuracy {ad['ood_accuracy']:.3f} was actually "
                f"*higher* than in-domain {ad['in_domain_accuracy']:.3f} (gap {gap:.3f}). "
                "Distance-based applicability domain did not act as a useful signal here, "
                "likely due to the small OOD sample size.\n"
            )
        else:
            lines.append(
                f"- **{ad['task']}**: no meaningful difference - OOD accuracy "
                f"{ad['ood_accuracy']:.3f} vs. in-domain {ad['in_domain_accuracy']:.3f} "
                f"(gap {gap:.3f}).\n"
            )

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index
    trained_at = datetime.now(timezone.utc).isoformat()

    results = [run_task(task, df, descriptors, trained_at) for task in TASKS]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)

    print(f"Wrote {RESULTS_PATH}\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
