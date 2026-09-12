"""False positive / false negative error analysis for the 15 classification
tasks (12 Tox21 assays + fda_approved + ct_tox + bbbp_penetration), per
TODO/ml/TODO_evaluation_validation.md's "Аналіз false positive / false
negative для toxicity-задач (які структурні класи молекул помиляються
найчастіше)".

Reloads the ACTUAL registered production models from
models/production/models.joblib (not fresh retrains) - same pattern as
ml/evaluate.py - and scores them on the val split (the split used for model
selection throughout this project; test stays untouched here since this is
diagnostic, not a final reported metric).

Two lenses on "which structural classes of molecules are over-represented
in the errors":

  1. Murcko scaffold (ml/preprocess.py's murcko_scaffold): for scaffolds that
     appear often enough in a task's val split to say anything (>= 5 rows),
     compare that scaffold group's error rate (FP+FN as a fraction of the
     group) against the task's overall val error rate. A scaffold group with
     a much higher relative error rate is "over-represented in errors" for
     that task.
  2. The same 7 physico-chemical descriptors used everywhere else in this
     project (ml/features.py's DESCRIPTOR_COLUMNS): compare descriptor means
     for false positives and false negatives against correctly-classified
     molecules, to see whether errors cluster in a particular descriptor
     region (very large/small molecules, extreme LogP, etc).

This is an honest, exploratory pass, not a hunt for a story: many tasks will
show nothing structurally conclusive (too few errors, too few molecules per
scaffold, or descriptor differences within noise) - the report says so
plainly rather than overstating a marginal difference.

Usage:
    python ml/error_analysis_toxicity.py

Reads models/production/models.joblib + data/processed/admet_processed.csv.
Writes ml/results/error_analysis_report.md.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    compute_descriptors,
    compute_fingerprints,
    task_rows,
)
from model_registry import feature_type_for
from preprocess import murcko_scaffold

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
PRODUCTION_DIR = ROOT / "models" / "production"
RESULTS_DIR = ROOT / "ml" / "results"
REPORT_PATH = RESULTS_DIR / "error_analysis_report.md"

DECISION_THRESHOLD = 0.5  # Plain 0.5 cutoff - this analysis is about which
# molecules get flipped to the wrong side of the decision boundary at the
# threshold the model would use by default, not a search for an optimal
# operating point (that is a separate, threshold-tuning question - see the
# note in TODO/ml/TODO_evaluation_validation.md about FP/FN cost trade-off).

MIN_SCAFFOLD_GROUP = 5  # below this, an "error rate" for the group is noise
SCAFFOLD_OVERREP_RATIO = 1.5  # group error rate >= this x task's overall error rate
DESCRIPTOR_FLAG_Z = 0.5  # |mean(errors) - mean(correct)| >= this many std devs of "correct" group


def load_val_predictions(df: pd.DataFrame, descriptors: pd.DataFrame, fingerprints: np.ndarray, bundle: dict, task: str):
    """Reuses the exact reload pattern from ml/evaluate.py: pull the
    registered bundle for this task and score it on the val split."""
    task_bundles = bundle["tasks"]
    if task not in task_bundles:
        return None
    info = task_bundles[task]
    model_name = info["model_name"]

    rows = task_rows(df, task)
    row_positions = df.index.get_indexer(rows.index)
    desc = descriptors.loc[rows.index, DESCRIPTOR_COLUMNS].to_numpy()
    fps = fingerprints[row_positions]
    X = fps if feature_type_for(model_name) == "fingerprints" else desc

    val_mask = (rows["split"] == "val").to_numpy()
    if val_mask.sum() == 0:
        return None

    y_val = rows[task].to_numpy().astype(int)[val_mask]
    model = info["model"]
    proba_val = model.predict_proba(X[val_mask])[:, 1]
    pred_val = (proba_val >= DECISION_THRESHOLD).astype(int)

    val_rows = rows.loc[val_mask]
    val_descriptors = descriptors.loc[val_rows.index, DESCRIPTOR_COLUMNS]

    return {
        "model_name": model_name,
        "smiles": val_rows["canonical_smiles"].to_numpy(),
        "y_true": y_val,
        "y_pred": pred_val,
        "proba": proba_val,
        "descriptors": val_descriptors.reset_index(drop=True),
    }


def scaffold_error_table(result: dict, scaffold_lookup: dict) -> pd.DataFrame:
    smiles = result["smiles"]
    y_true, y_pred = result["y_true"], result["y_pred"]
    is_error = y_true != y_pred
    is_fp = (y_true == 0) & (y_pred == 1)
    is_fn = (y_true == 1) & (y_pred == 0)
    scaffolds = [scaffold_lookup[s] for s in smiles]

    frame = pd.DataFrame({"scaffold": scaffolds, "is_error": is_error, "is_fp": is_fp, "is_fn": is_fn})
    grouped = frame.groupby("scaffold").agg(n=("is_error", "size"), n_error=("is_error", "sum"), n_fp=("is_fp", "sum"), n_fn=("is_fn", "sum"))
    grouped["error_rate"] = grouped["n_error"] / grouped["n"]
    return grouped.sort_values("n", ascending=False)


def descriptor_shift(result: dict) -> dict:
    """Mean/std of each descriptor for correctly-classified molecules vs. for
    false positives and false negatives separately. Returns flagged
    descriptors where the error-group mean differs from the correct-group
    mean by >= DESCRIPTOR_FLAG_Z standard deviations of the correct group."""
    y_true, y_pred = result["y_true"], result["y_pred"]
    desc = result["descriptors"]
    correct = (y_true == y_pred)
    is_fp = (y_true == 0) & (y_pred == 1)
    is_fn = (y_true == 1) & (y_pred == 0)

    out = {"n_correct": int(correct.sum()), "n_fp": int(is_fp.sum()), "n_fn": int(is_fn.sum()), "flags": []}
    if correct.sum() < 5:
        return out

    correct_mean = desc.loc[correct].mean()
    correct_std = desc.loc[correct].std().replace(0, np.nan)

    for label, mask in [("FP", is_fp), ("FN", is_fn)]:
        if mask.sum() < 5:
            continue
        group_mean = desc.loc[mask].mean()
        z = (group_mean - correct_mean) / correct_std
        for col in DESCRIPTOR_COLUMNS:
            if pd.isna(z[col]):
                continue
            if abs(z[col]) >= DESCRIPTOR_FLAG_Z:
                out["flags"].append(
                    {
                        "group": label,
                        "descriptor": col,
                        "group_mean": float(group_mean[col]),
                        "correct_mean": float(correct_mean[col]),
                        "z": float(z[col]),
                    }
                )
    return out


def analyze_task(df: pd.DataFrame, descriptors: pd.DataFrame, fingerprints: np.ndarray, bundle: dict, task: str, scaffold_lookup: dict) -> dict | None:
    result = load_val_predictions(df, descriptors, fingerprints, bundle, task)
    if result is None:
        return None

    y_true, y_pred = result["y_true"], result["y_pred"]
    n = len(y_true)
    n_fp = int(((y_true == 0) & (y_pred == 1)).sum())
    n_fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n_error = n_fp + n_fn
    overall_error_rate = n_error / n if n else 0.0

    scaffold_table = scaffold_error_table(result, scaffold_lookup)
    overrepresented = scaffold_table[
        (scaffold_table["n"] >= MIN_SCAFFOLD_GROUP) & (scaffold_table["error_rate"] >= SCAFFOLD_OVERREP_RATIO * max(overall_error_rate, 1e-6))
    ].sort_values("error_rate", ascending=False)

    desc_shift = descriptor_shift(result)

    return {
        "task": task,
        "model_name": result["model_name"],
        "n": n,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "overall_error_rate": overall_error_rate,
        "scaffold_groups_considered": int((scaffold_table["n"] >= MIN_SCAFFOLD_GROUP).sum()),
        "overrepresented_scaffolds": overrepresented,
        "descriptor_shift": desc_shift,
    }


def _fmt_scaffold(s: str, max_len: int = 40) -> str:
    if s == "":
        return "*(acyclic - no ring scaffold)*"
    s_escaped = s.replace("|", "\\|")
    return s_escaped if len(s_escaped) <= max_len else s_escaped[: max_len - 3] + "..."


def write_report(analyses: list[dict]) -> None:
    lines = [
        "# Error Analysis: False Positives / False Negatives (Toxicity + ADMET Classification Tasks)",
        "",
        (
            "Val-split predictions from the ACTUAL registered production models "
            "(`models/production/models.joblib`, reloaded the same way as "
            "`ml/evaluate.py` - not freshly retrained), at a plain 0.5 decision "
            f"threshold. {len(analyses)} of 15 classification tasks scored."
        ),
        "",
        (
            "Two lenses: (1) Murcko scaffold groups (via `ml/preprocess.py`'s "
            "`murcko_scaffold`) with an error rate notably above the task's overall "
            f"error rate (>= {SCAFFOLD_OVERREP_RATIO}x, group size >= {MIN_SCAFFOLD_GROUP} molecules "
            "in val); (2) whether the 7 shared physico-chemical descriptors "
            "(`ml/features.py`'s `DESCRIPTOR_COLUMNS`) differ between error groups "
            f"(FP/FN) and correctly-classified molecules by >= {DESCRIPTOR_FLAG_Z} standard "
            "deviations of the correct group's spread."
        ),
        "",
        (
            "This is exploratory and reported honestly: several tasks show no scaffold group "
            "or descriptor pattern that clears these (fairly loose) thresholds - that is stated "
            "plainly below rather than stretched into a finding."
        ),
        "",
        "## Cross-task pattern worth flagging",
        "",
        (
            "Nearly every Tox21 task below is FN-heavy (false negatives far outnumber false "
            "positives, often 0 FP at all) - a direct consequence of these tasks' low positive "
            "prevalence plus a plain 0.5 threshold: the calibrated models learn to require strong "
            "evidence before predicting the rare positive class, so they miss real positives more "
            "often than they falsely flag negatives. This is a threshold/calibration artifact, not "
            "evidence the models ignore structure - see the caveats below."
        ),
        (
            "The one recurring structural signal that shows up across several *different* nuclear "
            "receptor assays (NR-AR, NR-AR-LBD, NR-ER, NR-ER-LBD, and to a smaller extent "
            "bbbp_penetration) is the steroid-ketone scaffold `O=C1C=C2CCC3C4CCCC4CCC3C2CC1` "
            "(an androstenedione/steroid-hormone-like core): it is a small group in val (24-33 "
            "molecules) but its error rate is 6-24x the task's overall error rate, almost entirely "
            "false negatives. This makes biological sense - these four assays specifically test "
            "binding to steroid hormone receptors (androgen/estrogen), so real steroid-scaffold "
            "agonists/antagonists are exactly the molecules the model most needs to recognize, and "
            "it is instead systematically missing them. This is the one finding in this report that "
            "looks like a genuine, actionable structural blind spot rather than noise."
        ),
        (
            "A second, weaker recurring signal: the benzanilide-like scaffold "
            "`O=C(Nc1ccccc1)c1ccccc1` and the stilbene-like scaffold `C(=Cc1ccccc1)c1ccccc1` "
            "reappear as over-represented FN groups across several unrelated SR-* stress-response "
            "assays (SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53) - each individually a small group "
            "(7-20 molecules), so treat this as suggestive rather than conclusive."
        ),
        (
            "For fda_approved, ct_tox, and bbbp_penetration - the 3 non-Tox21 tasks with much "
            "smaller val sets (260-269 rows) - no scaffold group has enough repeated molecules to "
            "say anything structural; their descriptor shifts (e.g. higher LogP/RingCount in "
            "fda_approved's false positives) are the only signal available and are modest (z just "
            "above the 0.5 threshold)."
        ),
        "",
        "## Summary table",
        "",
        "| task | model | n (val) | FP | FN | error rate | scaffold groups considered | over-represented scaffolds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in analyses:
        n_over = len(a["overrepresented_scaffolds"])
        lines.append(
            f"| {a['task']} | {a['model_name']} | {a['n']} | {a['n_fp']} | {a['n_fn']} | "
            f"{a['overall_error_rate']:.3f} | {a['scaffold_groups_considered']} | {n_over} |"
        )

    lines += ["", "## Per-task detail", ""]
    for a in analyses:
        lines.append(f"### {a['task']} (model: {a['model_name']})")
        lines.append("")
        lines.append(
            f"n_val={a['n']}, FP={a['n_fp']}, FN={a['n_fn']}, overall error rate={a['overall_error_rate']:.3f}, "
            f"{a['scaffold_groups_considered']} scaffold group(s) with >= {MIN_SCAFFOLD_GROUP} molecules."
        )
        lines.append("")

        over = a["overrepresented_scaffolds"]
        if len(over) == 0:
            lines.append(
                "**Scaffold analysis:** no scaffold group reaches the over-representation "
                f"threshold ({SCAFFOLD_OVERREP_RATIO}x the task's overall error rate at n>={MIN_SCAFFOLD_GROUP}) - "
                "errors here do not concentrate in any one structural scaffold class large "
                "enough to say anything about; they look spread across the val set."
            )
        else:
            lines.append("**Scaffold analysis:** over-represented scaffold group(s):")
            lines.append("")
            lines.append("| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |")
            lines.append("|---|---|---|---|---|---|")
            for scaffold, row in over.iterrows():
                ratio = row["error_rate"] / max(a["overall_error_rate"], 1e-6)
                lines.append(
                    f"| {_fmt_scaffold(scaffold)} | {int(row['n'])} | {int(row['n_fp'])} | {int(row['n_fn'])} | "
                    f"{row['error_rate']:.3f} | {ratio:.1f}x |"
                )
        lines.append("")

        flags = a["descriptor_shift"]["flags"]
        if not flags:
            lines.append(
                "**Descriptor analysis:** no descriptor differs from the correctly-classified "
                f"group by >= {DESCRIPTOR_FLAG_Z} std devs for either FP or FN - errors do not "
                "obviously cluster in a particular size/LogP/polarity region for this task."
            )
        else:
            lines.append(f"**Descriptor analysis:** notable shift(s) (>= {DESCRIPTOR_FLAG_Z:.1f} std dev vs correctly-classified molecules):")
            lines.append("")
            lines.append("| group | descriptor | group mean | correct-group mean | z |")
            lines.append("|---|---|---|---|---|")
            for f in flags:
                lines.append(f"| {f['group']} | {f['descriptor']} | {f['group_mean']:.2f} | {f['correct_mean']:.2f} | {f['z']:+.2f} |")
        lines.append("")

    lines += [
        "## Caveats",
        "",
        (
            "- 0.5 is a default decision threshold, not a tuned operating point - FP/FN counts "
            "here would shift with a different threshold, and (per "
            "`TODO/ml/TODO_evaluation_validation.md`) false negatives and false positives likely "
            "have different real-world costs for a toxicity task that this analysis does not weigh."
        ),
        (
            "- Many val sets have only a handful of scaffold groups with >= 5 molecules (most "
            "scaffolds in this dataset are singletons - see the summary table's "
            "\"scaffold groups considered\" column) - absence of a flagged scaffold group often "
            "means there just isn't enough repetition of any one scaffold to say something "
            "about it, not that scaffold has no effect."
        ),
        (
            "- Descriptor z-scores are computed against the correctly-classified group's own "
            "spread per task, so they are not comparable in absolute terms across tasks."
        ),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    descriptors = compute_descriptors(df["canonical_smiles"])
    descriptors.index = df.index
    fingerprints = compute_fingerprints(df["canonical_smiles"])
    bundle = joblib.load(PRODUCTION_DIR / "models.joblib")

    scaffold_lookup = {s: murcko_scaffold(s) for s in df["canonical_smiles"].unique()}

    analyses = []
    for task in CLASSIFICATION_TASKS:
        result = analyze_task(df, descriptors, fingerprints, bundle, task, scaffold_lookup)
        if result is not None:
            analyses.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_report(analyses)

    print(f"Analyzed {len(analyses)}/{len(CLASSIFICATION_TASKS)} classification tasks.")
    for a in analyses:
        n_over = len(a["overrepresented_scaffolds"])
        n_flags = len(a["descriptor_shift"]["flags"])
        print(f"  {a['task']}: FP={a['n_fp']} FN={a['n_fn']} error_rate={a['overall_error_rate']:.3f} over-rep scaffolds={n_over} descriptor flags={n_flags}")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
