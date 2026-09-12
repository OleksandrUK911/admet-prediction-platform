"""Consolidated experiment-tracking report: aggregates results ALREADY
written by ml/baseline.py, ml/experiments_per_task_models.py,
ml/experiments_multitask.py, ml/model_registry.py, ml/evaluate.py and
ml/calibration.py into one per-task comparison table (baseline vs
per-task-tuned XGBoost vs multi-task NN vs the registered production
winner), plus artifact size and inference latency where a production
model bundle is available.

This does NOT retrain anything - it is a pure aggregation/reporting layer
over files that already exist in ml/results/ and models/production/.

## Why no MLflow / Weights & Biases here (deliberate scope decision)

Full experiment-tracking infrastructure (MLflow server, W&B project, a
run database, a model registry UI) is explicitly OUT OF SCOPE for this
portfolio-scale project - not because it's hard, but because it would be
disproportionate to the problem size: 16 tasks, a handful of candidate
model families per task, all run locally in minutes. Every experiment in
this project already writes deterministic, git-tracked JSON metrics +
Markdown reports to ml/results/ (baseline_metrics.json,
per_task_xgboost_metrics.json, multitask_metrics.json,
cross_validation_metrics.json, calibration_metrics.json) plus a
git-committed models/production/metadata.json for the registered winners.
That gives every property a heavier tracker would add at this scale -
reproducibility (same repo, same seed, same commit -> same numbers),
comparability across runs (diff two JSON files), and a durable audit
trail (git log) - without standing up and maintaining a separate service.
This script is the "comparison dashboard" that infra would otherwise
provide, built directly on those files.

## Winner-selection criterion (formal statement)

The FORMAL, current criterion for "production winner" per task is
implemented in `ml/model_registry.py`'s `pick_winner()` and is:

  - Classification tasks: the candidate (among
    logistic_regression_fingerprints, random_forest_descriptors,
    xgboost_tuned - i.e. NOT the majority_class/naive_mean floor models)
    with the highest **val PR-AUC** (not ROC-AUC - these are imbalanced
    tasks, see ml/results/per_task_comparison_report.md for why PR-AUC is
    the honest signal here).
  - Regression tasks: the candidate (among ridge_descriptors,
    xgboost_tuned) with the lowest **val RMSE**.
  - The multi-task NN (ml/experiments_multitask.py) is deliberately NOT
    a candidate in this selection - it is compared here for information
    (see the "multitask wins?" column below and
    ml/results/multitask_comparison_report.md) but the registry's
    winner-picking only chooses among independently-trained per-task
    baseline/XGBoost models. See "Known limitations" below for why.

Usage:
    python ml/experiments_tracking_summary.py

Reads (all optional - the script degrades gracefully if a file is
missing rather than crashing):
    ml/results/baseline_metrics.json
    ml/results/per_task_xgboost_metrics.json
    ml/results/multitask_metrics.json
    ml/results/cross_validation_metrics.json
    models/production/metadata.json
    models/production/models.joblib (gitignored, ~125MB - only present
        after a local `python ml/model_registry.py` run; artifact-size
        and latency benchmarking are skipped, with a clear note, when
        it's absent - regenerating it just for this report would mean
        re-running the multi-hour data pipeline, which is out of scope
        for a reporting script)
    data/processed/admet_processed.csv (for the latency micro-benchmark;
        same graceful skip if absent)

Writes ml/results/experiments_tracking_summary.md and
ml/results/experiments_tracking_summary.json. If models.joblib and the
processed CSV are both available it also writes a small trade-off
scatter plot to ml/results/tracking_quality_vs_latency.png (quality
metric vs. inference latency per task) - matplotlib is an optional
import; the script still runs and skips only that plot if matplotlib
isn't installed.
"""

import io
import json
import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CLASSIFICATION_TASKS, DESCRIPTOR_COLUMNS, REGRESSION_TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "ml" / "results"
PRODUCTION_DIR = ROOT / "models" / "production"
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"

BASELINE_PATH = RESULTS_DIR / "baseline_metrics.json"
PER_TASK_XGB_PATH = RESULTS_DIR / "per_task_xgboost_metrics.json"
MULTITASK_PATH = RESULTS_DIR / "multitask_metrics.json"
CV_PATH = RESULTS_DIR / "cross_validation_metrics.json"
METADATA_PATH = PRODUCTION_DIR / "metadata.json"
MODELS_JOBLIB_PATH = PRODUCTION_DIR / "models.joblib"

OUT_MD = RESULTS_DIR / "experiments_tracking_summary.md"
OUT_JSON = RESULTS_DIR / "experiments_tracking_summary.json"
OUT_PLOT = RESULTS_DIR / "tracking_quality_vs_latency.png"

# "Classic baseline" candidates - excludes the majority_class/naive_mean
# floor AND excludes xgboost_tuned (that's the "per-task-tuned" column).
BASELINE_CLASSIFICATION_CANDIDATES = ["logistic_regression_fingerprints", "random_forest_descriptors"]
BASELINE_REGRESSION_CANDIDATES = ["ridge_descriptors"]

N_LATENCY_SAMPLES = 50  # rows used for the inference micro-benchmark
N_LATENCY_REPEATS = 20  # timeit repeats per task


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_val_record(records: list[dict] | None, task: str, candidates: list[str], metric: str, higher_is_better: bool):
    if not records:
        return None
    scored = [
        r for r in records
        if r["task"] == task and r["split"] == "val" and r["model_name"] in candidates and r.get(metric) is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda r: r[metric]) if higher_is_better else min(scored, key=lambda r: r[metric])


def multitask_val_record(records: list[dict] | None, task: str):
    if not records:
        return None
    matches = [r for r in records if r["task"] == task and r["split"] == "val"]
    return matches[0] if matches else None


def build_comparison_table() -> list[dict]:
    baseline_records = load_json(BASELINE_PATH)
    per_task_records = load_json(PER_TASK_XGB_PATH)
    multitask_records = load_json(MULTITASK_PATH)
    metadata = load_json(METADATA_PATH)
    winners = (metadata or {}).get("per_task_winners", {})

    rows = []
    for task in REGRESSION_TASKS + CLASSIFICATION_TASKS:
        is_classification = task in CLASSIFICATION_TASKS
        metric = "pr_auc" if is_classification else "rmse"
        higher_is_better = is_classification

        baseline = best_val_record(
            baseline_records, task,
            BASELINE_CLASSIFICATION_CANDIDATES if is_classification else BASELINE_REGRESSION_CANDIDATES,
            metric, higher_is_better,
        )
        per_task_tuned = best_val_record(per_task_records, task, ["xgboost_tuned"], metric, higher_is_better)
        multitask = multitask_val_record(multitask_records, task)
        winner = winners.get(task)

        rows.append({
            "task": task,
            "task_type": "classification" if is_classification else "regression",
            "metric": metric,
            "baseline_model": baseline["model_name"] if baseline else None,
            "baseline_val": baseline[metric] if baseline else None,
            "per_task_tuned_val": per_task_tuned[metric] if per_task_tuned else None,
            "multitask_val": multitask[metric] if multitask else None,
            "registered_winner_model": winner["model_name"] if winner else None,
            "registered_winner_val": winner["val"].get(metric) if winner else None,
        })
    return rows


def try_load_bundle():
    """Returns (bundle, descriptors, fingerprints, df) if the production
    model artifact and processed dataset are both present locally, else
    None - this environment may only have the committed, small
    metadata.json (models.joblib and data/processed/ are gitignored,
    regenerated artifacts)."""
    if not MODELS_JOBLIB_PATH.exists() or not PROCESSED_CSV.exists():
        return None
    import joblib
    import pandas as pd

    from features import compute_descriptors, compute_fingerprints

    bundle = joblib.load(MODELS_JOBLIB_PATH)
    df = pd.read_csv(PROCESSED_CSV)
    sample_df = df.sample(n=min(N_LATENCY_SAMPLES, len(df)), random_state=42)
    descriptors = compute_descriptors(sample_df["canonical_smiles"])[DESCRIPTOR_COLUMNS].to_numpy()
    fingerprints = compute_fingerprints(sample_df["canonical_smiles"])
    return bundle, descriptors, fingerprints


def measure_artifact_sizes_and_latency(rows: list[dict]) -> None:
    """Mutates each row in-place, adding artifact_size_bytes and
    latency_ms_per_prediction where measurable."""
    for row in rows:
        row["artifact_size_bytes"] = None
        row["latency_ms_per_prediction"] = None

    loaded = try_load_bundle()
    if loaded is None:
        return
    bundle, descriptors, fingerprints = loaded
    task_bundles = bundle.get("tasks", {})
    by_task = {r["task"]: r for r in rows}

    import joblib

    for task, info in task_bundles.items():
        row = by_task.get(task)
        if row is None:
            continue

        # Artifact size: pickle just this task's bundle in-memory (not the
        # combined 125MB file) to get a genuinely per-task size.
        buf = io.BytesIO()
        joblib.dump(info, buf)
        row["artifact_size_bytes"] = buf.tell()

        # Latency: timeit a single BATCHED predict/predict_proba call over
        # N_LATENCY_SAMPLES rows (not a per-row Python loop - some estimators
        # here, e.g. RandomForest with n_jobs=-1, pay a real per-call
        # parallel-backend dispatch cost, which a tight per-row loop would
        # mostly measure instead of actual model compute). Dividing the
        # batched wall-clock time by the row count is a reasonable simple
        # micro-benchmark approximation of per-prediction latency, not a
        # claim about single-request (batch size 1) serving latency.
        feature_type = info["feature_type"]
        X = fingerprints if feature_type == "fingerprints" else descriptors
        model = info["model"]
        scaler = info.get("scaler")

        if info["task_type"] == "classification":
            def call(model=model, X=X):
                model.predict_proba(X)
        else:
            X_in = scaler.transform(X) if scaler is not None else X

            def call(model=model, X_in=X_in):
                model.predict(X_in)

        n_rows = X.shape[0]
        total_seconds = timeit.timeit(call, number=N_LATENCY_REPEATS)
        row["latency_ms_per_prediction"] = 1000.0 * total_seconds / (N_LATENCY_REPEATS * n_rows)


def format_metric(value, metric: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def write_report(rows: list[dict]) -> None:
    lines = ["# Consolidated Experiment-Tracking Summary\n\n"]
    lines.append(
        "Aggregates results already written by ml/baseline.py, "
        "ml/experiments_per_task_models.py, ml/experiments_multitask.py, "
        "ml/model_registry.py and ml/evaluate.py - nothing here is "
        "retrained. See this file's generating script "
        "(ml/experiments_tracking_summary.py) module docstring for the "
        "full scope-decision note on why MLflow/W&B is intentionally not "
        "used, and the formal winner-selection criterion.\n\n"
    )

    lines.append(
        "## Per-task comparison: baseline vs per-task-tuned vs multi-task vs registered winner\n\n"
        "Classification tasks are compared on val PR-AUC (higher is better); "
        "the one regression task (solubility) on val RMSE (lower is better). "
        "\"Baseline\" is the best of logistic-regression/random-forest "
        "(classification) or ridge (regression) - i.e. NOT the majority/naive "
        "floor and NOT XGBoost. \"Registered winner\" is whichever candidate "
        "`ml/model_registry.py`'s `pick_winner()` actually selected for "
        "production (multi-task NN is never a candidate there - see the "
        "criterion note above).\n\n"
        "| Task | Metric | Baseline (model) | Per-task-tuned XGBoost | Multi-task NN | "
        "Registered winner (model) |\n"
        "|---|---|---|---|---|---|\n"
    )
    for r in rows:
        baseline_str = f"{format_metric(r['baseline_val'], r['metric'])} ({r['baseline_model']})" if r["baseline_model"] else "n/a"
        winner_str = (
            f"{format_metric(r['registered_winner_val'], r['metric'])} ({r['registered_winner_model']})"
            if r["registered_winner_model"] else "not registered"
        )
        lines.append(
            f"| {r['task']} | {r['metric']} | {baseline_str} | "
            f"{format_metric(r['per_task_tuned_val'], r['metric'])} | "
            f"{format_metric(r['multitask_val'], r['metric'])} | {winner_str} |\n"
        )

    lines.append("\n## Artifact size and inference latency, per task\n\n")
    have_bench = any(r.get("artifact_size_bytes") is not None for r in rows)
    if not have_bench:
        lines.append(
            "Not measured in this run: `models/production/models.joblib` and/or "
            "`data/processed/admet_processed.csv` are not present locally (both "
            "are gitignored, regenerated artifacts - see `.gitignore`). This "
            "section is populated automatically whenever this script runs after "
            "`ml/model_registry.py` has produced a local `models.joblib` (e.g. "
            "the CI `ml` job, or a full local pipeline run) - regenerating them "
            "just to build this report would mean re-running the multi-hour "
            "data pipeline, which is out of scope here.\n"
        )
    else:
        lines.append(
            f"Latency = mean wall-clock time for one single-molecule prediction "
            f"call, timeit-measured over {N_LATENCY_SAMPLES} sampled rows x "
            f"{N_LATENCY_REPEATS} repeats (a simple micro-benchmark, not a "
            "production load test). Artifact size = bytes of that task's model "
            "bundle alone, pickled in-memory (not the combined 125MB file).\n\n"
            "| Task | Model | Artifact size | Latency (ms/prediction) |\n|---|---|---|---|\n"
        )
        for r in rows:
            size = r.get("artifact_size_bytes")
            latency = r.get("latency_ms_per_prediction")
            size_str = f"{size / 1024:.1f} KB" if size is not None else "n/a"
            latency_str = f"{latency:.3f}" if latency is not None else "n/a"
            lines.append(f"| {r['task']} | {r['registered_winner_model'] or 'n/a'} | {size_str} | {latency_str} |\n")

    OUT_MD.write_text("".join(lines), encoding="utf-8")


def write_plot(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping tracking_quality_vs_latency.png")
        return

    plottable = [r for r in rows if r.get("latency_ms_per_prediction") is not None and r.get("registered_winner_val") is not None]
    if not plottable:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in plottable:
        quality = r["registered_winner_val"] if r["metric"] == "pr_auc" else -r["registered_winner_val"]
        ax.scatter(r["latency_ms_per_prediction"], quality, label=r["task"])
        ax.annotate(r["task"], (r["latency_ms_per_prediction"], quality), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Inference latency (ms/prediction)")
    ax.set_ylabel("Quality (val PR-AUC, or -RMSE for the regression task)")
    ax.set_title("Quality vs. inference cost trade-off, per task's registered winner")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=120)
    plt.close(fig)


def main() -> None:
    rows = build_comparison_table()
    measure_artifact_sizes_and_latency(rows)
    write_report(rows)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_plot(rows)

    print(f"Wrote {OUT_MD}\nWrote {OUT_JSON}")
    if any(r.get("artifact_size_bytes") is not None for r in rows):
        print(f"Wrote {OUT_PLOT}")
    for r in rows:
        print(
            f"  {r['task']}: baseline={format_metric(r['baseline_val'], r['metric'])} "
            f"per_task_tuned={format_metric(r['per_task_tuned_val'], r['metric'])} "
            f"multitask={format_metric(r['multitask_val'], r['metric'])} "
            f"winner={r['registered_winner_model']}"
        )


if __name__ == "__main__":
    main()
