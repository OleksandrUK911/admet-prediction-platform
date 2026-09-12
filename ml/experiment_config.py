"""Single source of truth: which hyperparameters/features/model type actually
produced the CURRENTLY registered winner for each of the 16 ADMET tasks, per
TODO/ml/TODO_experiments_per_task_models.md's "Уніфікований конфіг per-task
навчання (гіперпараметри, фічі, split) для відтворюваності".

This module does NOT invent new hyperparameters. It consolidates what is
already scattered across:
  - ml/model_registry.py's `build_classifier()` / `retrain_regressor()` -
    the exact estimator + hyperparameters for each `model_name` (this is
    mirrored below as plain dicts; if `ml/model_registry.py` changes those
    hyperparameters, update MODEL_HYPERPARAMETERS here in the same commit -
    a `check_in_sync()` self-check at the bottom flags common drift).
  - ml/preprocess.py's scaffold split (ratios/seed) and ml/features.py's
    descriptor/fingerprint definitions.
  - models/production/metadata.json - which `model_name` actually won each
    task (read live, not copied by hand, so this file cannot go stale on
    which task uses which model_name after a re-run of
    ml/model_registry.py).

Reproducing a specific task's training run from this file:
    1. Look up TASK_CONFIG[task] (via load_task_config()) for its
       model_name, feature_type, and hyperparameters.
    2. Build features with ml/features.py's compute_descriptors /
       compute_fingerprints (feature_type tells you which).
    3. Split with ml/preprocess.py's scaffold_split (see SPLIT_CONFIG).
    4. Construct the estimator with MODEL_HYPERPARAMETERS[model_name] (same
       shape as ml/model_registry.py's build_classifier/retrain_regressor -
       for classification, wrap the fitted estimator in
       CalibratedClassifierCV per CALIBRATION_CONFIG).

Usage:
    python ml/experiment_config.py

Reads models/production/metadata.json.
Writes ml/experiment_config.json (versioned snapshot of the config below,
for a stable reference that does not require importing this module).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CLASSIFICATION_TASKS, DESCRIPTOR_COLUMNS, REGRESSION_TASKS

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_METADATA = ROOT / "models" / "production" / "metadata.json"
CONFIG_JSON_PATH = ROOT / "ml" / "experiment_config.json"

CONFIG_VERSION = "1.0.0"  # bump when the SHAPE of this config changes (new
# fields, restructuring) - not for every hyperparameter tweak (those are
# already versioned implicitly by models/production/metadata.json's
# model_version + trained_at, which this config is generated against).

SEED = 42

# --- Data split (ml/preprocess.py) ------------------------------------
SPLIT_CONFIG = {
    "method": "scaffold_split",
    "script": "ml/preprocess.py",
    "function": "scaffold_split",
    "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "seed": SEED,
    "notes": (
        "Groups molecules by Murcko scaffold (ml/preprocess.py's murcko_scaffold) "
        "before splitting, so structurally similar molecules land in the same split - "
        "a stricter, more realistic test of generalization than a random row split. "
        "Not to be confused with the per-scaffold error-rate analysis in "
        "ml/error_analysis_toxicity.py, which reads this same split's val rows but does "
        "not change it."
    ),
}

# --- Features (ml/features.py) -----------------------------------------
FEATURES_CONFIG = {
    "descriptors": {
        "columns": DESCRIPTOR_COLUMNS,
        "function": "ml/features.py:compute_descriptors",
        "source": "RDKit Descriptors (same 7 physico-chemical descriptors as project #1, for portfolio consistency)",
    },
    "fingerprints": {
        "type": "Morgan (ECFP-like)",
        "radius": 2,
        "n_bits": 1024,
        "function": "ml/features.py:compute_fingerprints",
    },
}

# --- Calibration (ml/model_registry.py) --------------------------------
CALIBRATION_CONFIG = {
    "applied_to": "every classification task's final registered model",
    "method": "Platt scaling (sigmoid) via sklearn CalibratedClassifierCV",
    "cv_folds": "3, or 2 if the minority class has fewer than 6 examples in the train split",
    "rationale": (
        "Platt scaling won 3/4 tasks explored in ml/calibration.py's isotonic-vs-Platt "
        "comparison (ml/results/calibration_report.md) - applied here as the default for "
        "all 15 classification tasks, not re-verified individually per task."
    ),
    "script": "ml/model_registry.py",
}

# --- Per-model-name hyperparameters, mirrored VERBATIM from
# ml/model_registry.py's build_classifier() / retrain_regressor() ---------
MODEL_HYPERPARAMETERS = {
    "logistic_regression_fingerprints": {
        "task_type": "classification",
        "feature_type": "fingerprints",
        "estimator": "sklearn.linear_model.LogisticRegression",
        "hyperparameters": {"max_iter": 1000, "random_state": SEED},
        "source": "ml/model_registry.py:build_classifier",
    },
    "random_forest_descriptors": {
        "task_type": "classification",
        "feature_type": "descriptors",
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "hyperparameters": {"n_estimators": 200, "random_state": SEED, "n_jobs": -1},
        "source": "ml/model_registry.py:build_classifier",
    },
    "xgboost_tuned": {
        # Two different task_types share this model_name - the classification
        # and regression hyperparameters are genuinely different (see
        # ml/model_registry.py's build_classifier vs retrain_regressor), so both
        # are recorded; which one applies depends on the task (see TASK_CONFIG).
        "classification": {
            # ml/model_registry.py's feature_type_for() only special-cases
            # "logistic_regression_fingerprints" as fingerprints - every other
            # model_name, xgboost_tuned included, trains on descriptors.
            "feature_type": "descriptors",
            "estimator": "xgboost.XGBClassifier",
            "hyperparameters": {
                "random_state": SEED,
                "n_jobs": -1,
                "scale_pos_weight": "train negatives / max(train positives, 1) - computed per task, not fixed",
                "eval_metric": "logloss",
            },
            "source": "ml/model_registry.py:build_classifier",
        },
        "regression": {
            "feature_type": "descriptors",
            "estimator": "xgboost.XGBRegressor",
            "hyperparameters": {"random_state": SEED, "n_jobs": -1, "max_depth": 3, "n_estimators": 300, "learning_rate": 0.05},
            "source": (
                "ml/model_registry.py:retrain_regressor (these exact values were selected as the "
                "best of a small fixed grid in ml/experiments_per_task_models.py's REGRESSION_GRID "
                "by val RMSE - see ml/results/per_task_xgboost_metrics.json - and then hardcoded "
                "into retrain_regressor as the production choice for solubility)."
            ),
        },
    },
    "ridge_descriptors": {
        "task_type": "regression",
        "feature_type": "descriptors",
        "estimator": "sklearn.linear_model.Ridge",
        "hyperparameters": {"alpha": 1.0, "random_state": SEED},
        "preprocessing": "sklearn.preprocessing.StandardScaler fit on the train split only",
        "source": "ml/model_registry.py:retrain_regressor",
    },
}


def load_task_config() -> dict:
    """Reads models/production/metadata.json (NOT hand-copied) for which
    model_name actually won each task, and joins it with the static
    hyperparameter/feature/calibration config above."""
    metadata = json.loads(PRODUCTION_METADATA.read_text(encoding="utf-8"))
    winners = metadata["per_task_winners"]

    task_config = {}
    for task in REGRESSION_TASKS + CLASSIFICATION_TASKS:
        if task not in winners:
            continue  # no registered winner for this task
        model_name = winners[task]["model_name"]
        is_classification = task in CLASSIFICATION_TASKS

        if model_name == "xgboost_tuned":
            variant = "classification" if is_classification else "regression"
            model_spec = MODEL_HYPERPARAMETERS["xgboost_tuned"][variant]
            feature_type = "fingerprints" if model_name == "logistic_regression_fingerprints" else "descriptors"
        else:
            model_spec = MODEL_HYPERPARAMETERS[model_name]
            feature_type = model_spec["feature_type"]

        entry = {
            "model_name": model_name,
            "task_type": "classification" if is_classification else "regression",
            "feature_type": feature_type,
            "hyperparameters": model_spec["hyperparameters"],
            "hyperparameters_source": model_spec["source"],
            "split": "train (fit) / val (model + hyperparameter selection) / test (final report only) - see SPLIT_CONFIG",
        }
        if is_classification:
            entry["calibration"] = CALIBRATION_CONFIG
        elif "preprocessing" in model_spec:
            entry["preprocessing"] = model_spec["preprocessing"]

        task_config[task] = entry
    return task_config


def check_in_sync() -> list[str]:
    """Cheap self-check: every model_name currently registered in
    models/production/metadata.json must have an entry here. Does not (and
    cannot, without re-importing sklearn/xgboost estimator internals)
    verify the hyperparameter VALUES stay byte-identical to
    ml/model_registry.py - that discipline is manual (see this module's
    docstring) - but it does catch the more common drift of a new
    model_name being introduced in ml/model_registry.py without a matching
    entry added here."""
    metadata = json.loads(PRODUCTION_METADATA.read_text(encoding="utf-8"))
    problems = []
    for task, info in metadata["per_task_winners"].items():
        model_name = info["model_name"]
        if model_name == "xgboost_tuned":
            continue  # handled specially (task_type-dependent variant), see load_task_config
        if model_name not in MODEL_HYPERPARAMETERS:
            problems.append(f"{task}: model_name '{model_name}' has no entry in MODEL_HYPERPARAMETERS")
    return problems


def main() -> None:
    problems = check_in_sync()
    if problems:
        print("WARNING - experiment_config.py may be out of sync with model_registry.py:")
        for p in problems:
            print(f"  - {p}")

    task_config = load_task_config()
    full_config = {
        "config_version": CONFIG_VERSION,
        "generated_from": "models/production/metadata.json + ml/model_registry.py (hyperparameters mirrored manually - see this module's docstring)",
        "seed": SEED,
        "split": SPLIT_CONFIG,
        "features": FEATURES_CONFIG,
        "calibration": CALIBRATION_CONFIG,
        "model_hyperparameters": MODEL_HYPERPARAMETERS,
        "tasks": task_config,
    }
    CONFIG_JSON_PATH.write_text(json.dumps(full_config, indent=2), encoding="utf-8")
    print(f"Wrote {CONFIG_JSON_PATH} ({len(task_config)} tasks documented).")


if __name__ == "__main__":
    main()
