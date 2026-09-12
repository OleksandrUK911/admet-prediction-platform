"""Feature engineering experiments and shared preprocessing artifacts.

Three pieces, per ml/TODO_feature_engineering.md's plan (not present in this
worktree, but the mandate is: extend ml/baseline.py's feature choices with a
bit more rigor before building anything more sophisticated on top of them):

1. Fingerprint radius/bit-size comparison on 3 representative tasks
   (NR-AR: small/imbalanced Tox21 assay, SR-MMP: larger/more-balanced Tox21
   assay, bbbp_penetration: small non-Tox21 task) - is the current
   radius=2/1024 choice in ml/features.py actually a good one, or would a
   larger radius/bit count meaningfully improve Logistic-Regression-on-
   fingerprints val ROC-AUC?
2. Descriptor-vs-label correlation across all 16 tasks, plus a
   multicollinearity check among the 7 descriptors themselves (replicating
   project #1's ESOL finding of TPSA correlating with NumHDonors/
   NumHAcceptors on this project's larger, more diverse molecule set).
3. A StandardScaler fit on the 7 descriptors (train split only) saved to
   ml/artifacts/descriptor_scaler.joblib for a future inference module to
   reuse - same bundle shape as project #1's descriptor_scaler.joblib
   ({"scaler": ..., "feature_names": [...]}).

Usage:
    python ml/feature_engineering.py

Only reads/imports from ml/features.py - does not modify it or ml/baseline.py.
"""

import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    compute_descriptors,
    compute_fingerprints,
    task_rows,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
RESULTS_DIR = ROOT / "ml" / "results"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
FP_REPORT_PATH = RESULTS_DIR / "fingerprint_size_comparison_report.md"
DESCRIPTOR_REPORT_PATH = RESULTS_DIR / "descriptor_relevance_report.md"
SCALER_PATH = ARTIFACTS_DIR / "descriptor_scaler.joblib"
SEED = 42

# --- Task 1: fingerprint radius/bit-size comparison -----------------------

REPRESENTATIVE_TASKS = ["NR-AR", "SR-MMP", "bbbp_penetration"]
FP_VARIANTS = [
    (2, 1024),  # current default in ml/features.py
    (3, 1024),
    (2, 2048),
    (3, 2048),
]
MEANINGFUL_DELTA = 0.02  # ROC-AUC gap below this is treated as noise


def compare_fingerprint_variants(df: pd.DataFrame) -> list[dict]:
    records = []
    for task in REPRESENTATIVE_TASKS:
        rows = task_rows(df, task)
        y = rows[task].to_numpy().astype(int)
        train_mask = (rows["split"] == "train").to_numpy()
        val_mask = (rows["split"] == "val").to_numpy()

        for radius, n_bits in FP_VARIANTS:
            fps = compute_fingerprints(rows["canonical_smiles"], radius=radius, n_bits=n_bits)
            lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(fps[train_mask], y[train_mask])
            proba = lr.predict_proba(fps[val_mask])[:, 1]
            auc = roc_auc_score(y[val_mask], proba) if len(np.unique(y[val_mask])) > 1 else None
            records.append({
                "task": task,
                "radius": radius,
                "n_bits": n_bits,
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "val_roc_auc": auc,
            })
    return records


def write_fingerprint_report(records: list[dict]) -> None:
    intro = (
        "Logistic Regression on Morgan fingerprints, val ROC-AUC, for 3 "
        "representative tasks (NR-AR: small/imbalanced Tox21 assay, SR-MMP: "
        "larger/more-balanced Tox21 assay, bbbp_penetration: small "
        f"non-Tox21 task). 'Meaningful' is defined as a val ROC-AUC "
        f"improvement of more than {MEANINGFUL_DELTA} over the current "
        "radius=2/1024-bit baseline used in ml/features.py - smaller gaps "
        "are treated as noise given val set sizes here.\n\n"
    )
    lines = [
        "# Fingerprint radius/bit-size comparison\n\n",
        intro,
        "| Task | Radius | Bits | N train | N val | Val ROC-AUC |\n",
        "|---|---|---|---|---|---|\n",
    ]
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)
        auc = f"{r['val_roc_auc']:.4f}" if r["val_roc_auc"] is not None else "n/a"
        marker = " (current default)" if (r["radius"], r["n_bits"]) == (2, 1024) else ""
        lines.append(
            f"| {r['task']} | {r['radius']} | {r['n_bits']} | {r['n_train']} | "
            f"{r['n_val']} | {auc}{marker} |\n"
        )

    lines.append("\n## Deltas vs. radius=2/1024 baseline\n\n")
    lines.append("| Task | Variant | Val ROC-AUC | Delta vs baseline | Meaningful? |\n")
    lines.append("|---|---|---|---|---|\n")
    verdicts = []
    for task, task_records in by_task.items():
        baseline = next(r for r in task_records if (r["radius"], r["n_bits"]) == (2, 1024))
        base_auc = baseline["val_roc_auc"]
        for r in task_records:
            if (r["radius"], r["n_bits"]) == (2, 1024):
                continue
            if base_auc is None or r["val_roc_auc"] is None:
                lines.append(f"| {task} | r={r['radius']}/{r['n_bits']}b | n/a | n/a | n/a |\n")
                continue
            delta = r["val_roc_auc"] - base_auc
            meaningful = abs(delta) > MEANINGFUL_DELTA
            verdicts.append(meaningful)
            lines.append(
                f"| {task} | r={r['radius']}/{r['n_bits']}b | {r['val_roc_auc']:.4f} | "
                f"{delta:+.4f} | {'YES' if meaningful else 'no (noise)'} |\n"
            )

    lines.append("\n## Verdict\n\n")
    if any(verdicts):
        lines.append(
            f"At least one variant crossed the {MEANINGFUL_DELTA} meaningful-delta "
            "threshold on at least one representative task - see the table above "
            "for which one(s). Given this is a single train/val split (not "
            "cross-validated), treat a single crossing with some caution before "
            "committing to a new default.\n"
        )
    else:
        lines.append(
            f"No variant beat the current radius=2/1024 default by more than "
            f"{MEANINGFUL_DELTA} val ROC-AUC on any of the 3 representative tasks. "
            "The differences observed are consistent with noise from a single "
            "train/val split, not a real effect. Recommendation: keep "
            "radius=2/1024 as the default in ml/features.py - it is not worth "
            "the extra compute/memory for 2048-bit fingerprints or the extra "
            "chemical-neighborhood radius given these results.\n"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FP_REPORT_PATH.write_text("".join(lines), encoding="utf-8")


# --- Task 2: descriptor relevance + multicollinearity ----------------------


def compute_descriptor_label_correlations(df: pd.DataFrame, descriptors: pd.DataFrame) -> dict[str, dict]:
    """Point-biserial/Pearson correlation between each descriptor and each
    task's label, using only task_rows() (non-missing-label rows). Point-
    biserial correlation is mathematically identical to Pearson correlation
    when one variable is binary, so this covers regression and
    classification tasks uniformly."""
    per_task = {}
    for task in ALL_TASKS:
        rows = task_rows(df, task)
        desc = descriptors.loc[rows.index]
        y = rows[task].to_numpy(dtype=float)
        corrs = {}
        for col in DESCRIPTOR_COLUMNS:
            x = desc[col].to_numpy(dtype=float)
            if np.std(x) == 0 or np.std(y) == 0:
                corrs[col] = float("nan")
                continue
            r, _p = pointbiserialr(y, x) if task in CLASSIFICATION_TASKS else (np.corrcoef(x, y)[0, 1], None)
            corrs[col] = float(r)
        per_task[task] = {"n": len(rows), "correlations": corrs}
    return per_task


def compute_descriptor_multicollinearity(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation among the 7 descriptors themselves,
    computed over all molecules (dataset-independent property of the
    descriptor definitions, not any one task's labels)."""
    return descriptors[DESCRIPTOR_COLUMNS].corr(method="pearson")


def write_descriptor_report(per_task_corrs: dict[str, dict], multicollinearity: pd.DataFrame) -> None:
    lines = ["# Descriptor relevance and multicollinearity report\n\n"]

    lines.append("## Descriptor-vs-label correlation, per task\n\n")
    lines.append(
        "Point-biserial correlation (= Pearson correlation with a binary "
        "variable) between each of the 7 descriptors and each task's label, "
        "computed on task_rows(df, task) - i.e. only rows with a non-missing "
        "label for that task. Top descriptor(s) are the ones with the "
        "largest |r|.\n\n"
    )
    lines.append("| Task | N | Top descriptor | r | 2nd descriptor | r |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for task, info in per_task_corrs.items():
        corrs = info["correlations"]
        ranked = sorted(corrs.items(), key=lambda kv: abs(kv[1]) if not np.isnan(kv[1]) else -1, reverse=True)
        top1, top1_r = ranked[0]
        top2, top2_r = ranked[1]
        lines.append(f"| {task} | {info['n']} | {top1} | {top1_r:.3f} | {top2} | {top2_r:.3f} |\n")

    lines.append("\n## Intuition check\n\n")
    bbbp = per_task_corrs["bbbp_penetration"]["correlations"]
    sol = per_task_corrs["solubility"]["correlations"]
    bbbp_top = max(bbbp.items(), key=lambda kv: abs(kv[1]))[0]
    sol_top = max(sol.items(), key=lambda kv: abs(kv[1]))[0]
    lines.append(
        f"- BBBP penetration: planning-doc intuition was TPSA/LogP should matter most. "
        f"Top descriptor found: **{bbbp_top}** (TPSA r={bbbp['TPSA']:.3f}, LogP r={bbbp['LogP']:.3f}). "
        f"{'Matches intuition.' if bbbp_top in ('TPSA', 'LogP') else 'Does NOT match intuition - TPSA/LogP were not the strongest correlate here.'}\n"
    )
    lines.append(
        f"- Solubility: planning-doc intuition was LogP/MolWt should matter most. "
        f"Top descriptor found: **{sol_top}** (LogP r={sol['LogP']:.3f}, MolWt r={sol['MolWt']:.3f}). "
        f"{'Matches intuition.' if sol_top in ('LogP', 'MolWt') else 'Does NOT match intuition - LogP/MolWt were not the strongest correlate here.'}\n"
    )
    lines.append(
        "- Tox21 assays / ClinTox: no prior intuition was assumed - correlations reported "
        "as-found above, without forcing a narrative.\n"
    )

    lines.append("\n## Descriptor-descriptor multicollinearity (all ~10130 molecules)\n\n")
    lines.append(
        "Project #1 found (on ESOL, 1128 molecules) that TPSA correlates strongly with "
        "NumHDonors (r=0.755) and NumHAcceptors (r=0.899). Recomputed here on this "
        "project's larger, more chemically diverse ~10130-molecule set:\n\n"
    )
    tpsa_hdonors = multicollinearity.loc["TPSA", "NumHDonors"]
    tpsa_hacceptors = multicollinearity.loc["TPSA", "NumHAcceptors"]
    lines.append(f"- TPSA vs NumHDonors: r = {tpsa_hdonors:.3f}\n")
    lines.append(f"- TPSA vs NumHAcceptors: r = {tpsa_hacceptors:.3f}\n")
    replicated = tpsa_hdonors > 0.5 and tpsa_hacceptors > 0.5
    lines.append(
        f"\n**{'Confirmed' if replicated else 'NOT confirmed'}**: the TPSA/H-bonding "
        "multicollinearity "
        f"{'holds' if replicated else 'does not clearly hold'} on this larger, more diverse "
        "dataset too - consistent with it being a property of the descriptors' formulas "
        "(TPSA is computed as a sum over polar-atom surface-area contributions that are "
        "closely related to H-bond donor/acceptor counts) rather than an artifact specific "
        "to ESOL.\n"
    )

    lines.append("\n### Full descriptor-descriptor correlation matrix\n\n")
    lines.append("| | " + " | ".join(DESCRIPTOR_COLUMNS) + " |\n")
    lines.append("|---" * (len(DESCRIPTOR_COLUMNS) + 1) + "|\n")
    for row_name in DESCRIPTOR_COLUMNS:
        vals = " | ".join(f"{multicollinearity.loc[row_name, col]:.3f}" for col in DESCRIPTOR_COLUMNS)
        lines.append(f"| {row_name} | {vals} |\n")

    lines.append("\n### Other notable descriptor pairs (|r| > 0.5, excluding the diagonal)\n\n")
    seen = set()
    notable = []
    for a, b in combinations(DESCRIPTOR_COLUMNS, 2):
        r = multicollinearity.loc[a, b]
        if abs(r) > 0.5 and (a, b) not in seen:
            notable.append((a, b, r))
            seen.add((a, b))
    if notable:
        lines.append("| Descriptor A | Descriptor B | r |\n|---|---|---|\n")
        for a, b, r in sorted(notable, key=lambda t: -abs(t[2])):
            lines.append(f"| {a} | {b} | {r:.3f} |\n")
    else:
        lines.append("None found.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DESCRIPTOR_REPORT_PATH.write_text("".join(lines), encoding="utf-8")


# --- Task 3: scaler fit on train split, persisted for inference ------------


def fit_and_save_descriptor_scaler(df: pd.DataFrame, descriptors: pd.DataFrame) -> StandardScaler:
    train_mask = (df["split"] == "train").to_numpy()
    X_train = descriptors.loc[df.index[train_mask], DESCRIPTOR_COLUMNS].to_numpy()
    X_val = descriptors.loc[df.index[~train_mask], DESCRIPTOR_COLUMNS].to_numpy()

    scaler = StandardScaler().fit(X_train)

    # Evidence the scaler was fit on train only: re-fitting on train+val
    # would shift the mean/scale if val had different statistics. Print a
    # quick comparison so this is visible at run time, not just asserted.
    combined_scaler = StandardScaler().fit(np.vstack([X_train, X_val]))
    mean_diff = np.abs(scaler.mean_ - combined_scaler.mean_).max()
    print(f"[scaler] fit on train only: n={len(X_train)}")
    print(f"[scaler] train-only mean: {np.round(scaler.mean_, 3)}")
    print(f"[scaler] train+val mean:  {np.round(combined_scaler.mean_, 3)}")
    print(f"[scaler] max |mean diff| train-only vs train+val: {mean_diff:.4f} "
          f"({'differs as expected - confirms train-only fit' if mean_diff > 1e-9 else 'WARNING: identical, check split'})")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": DESCRIPTOR_COLUMNS}, SCALER_PATH)
    print(f"Wrote {SCALER_PATH}")
    return scaler


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index

    print("Running fingerprint radius/bit-size comparison...")
    fp_records = compare_fingerprint_variants(df)
    write_fingerprint_report(fp_records)
    print(f"Wrote {FP_REPORT_PATH}")

    print("Computing descriptor relevance + multicollinearity...")
    per_task_corrs = compute_descriptor_label_correlations(df, descriptors)
    multicollinearity = compute_descriptor_multicollinearity(descriptors)
    write_descriptor_report(per_task_corrs, multicollinearity)
    print(f"Wrote {DESCRIPTOR_REPORT_PATH}")

    print("Fitting + saving descriptor scaler (train split only)...")
    fit_and_save_descriptor_scaler(df, descriptors)


if __name__ == "__main__":
    main()
