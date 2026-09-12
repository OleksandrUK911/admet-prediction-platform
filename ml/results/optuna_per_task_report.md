# Optuna per-task hyperparameter search vs. fixed-hyperparameter XGBoost

Tuned 4 of 15 classification tasks (a representative subset - well-balanced/`fda_approved`, two severely imbalanced/`SR-ATAD5` and `NR-PPAR-gamma`, and one mid-size win/`SR-MMP`), not all 15 - a full search over every classification task is expensive and out of scope here; each task ran 30 Optuna trials with 5-fold stratified CV PR-AUC (on TRAIN+VAL only) as the search objective. Full-coverage tuning across all 15 tasks remains future work (see TODO/ml/TODO_experiments_per_task_models.md).

`ml/experiments_per_task_models.py`'s fixed hyperparameters (`n_estimators=300, max_depth=4, learning_rate=0.05`) is the comparison point ("fixed" below), reported from `ml/results/per_task_xgboost_metrics.json`. Val PR-AUC decides whether tuning helped; test is reported for completeness only.

| Task | Split | Fixed PR-AUC | Optuna PR-AUC | Fixed ROC-AUC | Optuna ROC-AUC | Improved (val PR-AUC)? |
|---|---|---|---|---|---|---|
| fda_approved | val | 0.9753 | 0.9724 | 0.7134 | 0.6787 | No |
| fda_approved | test | 0.9213 | 0.9205 | 0.5724 | 0.5077 | No |
| SR-ATAD5 | val | 0.1234 | 0.1068 | 0.6724 | 0.6583 | No |
| SR-ATAD5 | test | 0.1410 | 0.1464 | 0.6833 | 0.6847 | Yes |
| NR-PPAR-gamma | val | 0.1021 | 0.1128 | 0.6798 | 0.7287 | Yes |
| NR-PPAR-gamma | test | 0.2067 | 0.2230 | 0.8216 | 0.7731 | Yes |
| SR-MMP | val | 0.5129 | 0.5431 | 0.7756 | 0.7786 | Yes |
| SR-MMP | test | 0.4802 | 0.5178 | 0.7584 | 0.7851 | Yes |

**Summary (vs. the fixed-hyperparameter XGBoost)**: Optuna-tuned XGBoost improved val PR-AUC over the currently fixed hyperparameters on 2 of 4 tuned tasks; it did not on 2.

## Comparison against the actually-registered production winner

The table above compares against `ml/experiments_per_task_models.py`'s own fixed-hyperparameter XGBoost (an apples-to-apples, same-model-family comparison). But XGBoost is not always the model `ml/model_registry.py` actually picked for a task - for 3 of these 4 tasks a baseline model (logistic regression or random forest) currently wins on val. The table below compares the Optuna-tuned XGBoost against whatever model is actually registered in `models/production/metadata.json` today, which is the more meaningful bar to clear.

| Task | Registered winner | Registered val PR-AUC | Optuna val PR-AUC | Beats registered winner? |
|---|---|---|---|---|
| fda_approved | logistic_regression_fingerprints | 0.9893 | 0.9724 | No |
| SR-ATAD5 | random_forest_descriptors | 0.1506 | 0.1068 | No |
| NR-PPAR-gamma | logistic_regression_fingerprints | 0.1151 | 0.1128 | No |
| SR-MMP | xgboost_tuned | 0.5137 | 0.5431 | Yes |

**Summary (vs. the registered production winner)**: the Optuna-tuned XGBoost beat the currently-registered winner's val PR-AUC on 1 of 4 tuned tasks. This is the honest bottom line for whether tuning would be worth promoting, and it is consistent with this project's established track record (per-task XGBoost only beat baseline on 3/15 tasks, multitask only beat per-task on 3/16): hyperparameter search does not change the fundamental conclusion that model/hyperparameter choice matters less than per-task data scarcity here. Regardless of outcome, no tuned model from this script is promoted into `ml/model_registry.py` - it is an exploratory comparison only, same status as `ml/feature_engineering.py`, `ml/calibration.py`, and `ml/experiments_multitask.py`.
