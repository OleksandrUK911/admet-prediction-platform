"""Reliability-diagram visualization for the 4 representative tasks
already explored in ml/calibration.py (NR-AR, SR-MMP, bbbp_penetration,
ct_tox).

ml/calibration.py computes Expected Calibration Error (a single number
per method) but never plots the underlying per-bin curve - this script
fills that specific, previously-noted gap (see
ml/TODO_calibration_uncertainty.md / ml/TODO_evaluation_validation.md:
"ECE (не reliability diagram-зображення)"). It reuses the exact same
model construction as ml/calibration.py (same RandomForest architecture,
same CalibratedClassifierCV settings, same val split, same SEED) so the
plotted curves are consistent with the ECE numbers already reported in
ml/results/calibration_report.md - it does not introduce a different
comparison.

For each task, plots predicted-probability bin midpoint vs. observed
positive rate in that bin (the standard reliability diagram), for raw RF,
Platt-calibrated, and isotonic-calibrated, against the y=x diagonal
(perfect calibration). Point size encodes bin population.

Usage:
    python ml/plot_reliability_diagrams.py

Reads data/processed/admet_processed.csv (must exist - run
ml/preprocess.py first if missing).
Writes ml/results/reliability_diagrams.png.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import DESCRIPTOR_COLUMNS, compute_descriptors, task_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
OUT_PATH = ROOT / "ml" / "results" / "reliability_diagrams.png"
SEED = 42
TASKS = ["NR-AR", "SR-MMP", "bbbp_penetration", "ct_tox"]
N_BINS = 10


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)


def binned_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS):
    """Returns (bin_midpoints, observed_rate, bin_counts) for populated bins
    only - same binning convention as ml/calibration.py's ECE."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    mids, rates, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        mids.append(float(y_prob[mask].mean()))
        rates.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(mids), np.array(rates), np.array(counts)


def run_task(task: str, df: pd.DataFrame, descriptors: pd.DataFrame) -> dict:
    rows = task_rows(df, task)
    desc = descriptors.loc[rows.index][DESCRIPTOR_COLUMNS].to_numpy()
    y = rows[task].to_numpy().astype(int)
    split = rows["split"].to_numpy()

    train_mask, val_mask = split == "train", split == "val"
    X_train, y_train = desc[train_mask], y[train_mask]
    X_val, y_val = desc[val_mask], y[val_mask]

    raw_rf = make_rf().fit(X_train, y_train)
    platt = CalibratedClassifierCV(make_rf(), method="sigmoid", cv=5).fit(X_train, y_train)
    isotonic = CalibratedClassifierCV(make_rf(), method="isotonic", cv=5).fit(X_train, y_train)

    curves = {}
    for name, model in [("raw_rf", raw_rf), ("platt_sigmoid", platt), ("isotonic", isotonic)]:
        proba = model.predict_proba(X_val)[:, 1]
        curves[name] = binned_curve(y_val, proba)
    return curves


def plot_all(all_curves: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    colors = {"raw_rf": "tab:gray", "platt_sigmoid": "tab:blue", "isotonic": "tab:orange"}

    for ax, task in zip(axes.flat, TASKS):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        for method, (mids, rates, counts) in all_curves[task].items():
            if len(mids) == 0:
                continue
            sizes = 20 + 200 * (counts / counts.max())
            ax.scatter(mids, rates, s=sizes, alpha=0.7, color=colors[method], label=method)
            ax.plot(mids, rates, color=colors[method], alpha=0.4, linewidth=1)
        ax.set_title(task)
        ax.set_xlabel("Mean predicted probability (bin)")
        ax.set_ylabel("Observed positive rate (bin)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Reliability diagrams (val split) - point size = bin population\n"
        "See ml/results/calibration_report.md for the corresponding ECE numbers",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=120)
    plt.close(fig)


def main() -> None:
    if not PROCESSED_CSV.exists():
        print(f"ERROR: {PROCESSED_CSV} not found - run ml/preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index

    all_curves = {task: run_task(task, df, descriptors) for task in TASKS}
    plot_all(all_curves)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
