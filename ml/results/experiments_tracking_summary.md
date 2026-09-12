# Consolidated Experiment-Tracking Summary

Aggregates results already written by ml/baseline.py, ml/experiments_per_task_models.py, ml/experiments_multitask.py, ml/model_registry.py and ml/evaluate.py - nothing here is retrained. See this file's generating script (ml/experiments_tracking_summary.py) module docstring for the full scope-decision note on why MLflow/W&B is intentionally not used, and the formal winner-selection criterion.

## Per-task comparison: baseline vs per-task-tuned vs multi-task vs registered winner

Classification tasks are compared on val PR-AUC (higher is better); the one regression task (solubility) on val RMSE (lower is better). "Baseline" is the best of logistic-regression/random-forest (classification) or ridge (regression) - i.e. NOT the majority/naive floor and NOT XGBoost. "Registered winner" is whichever candidate `ml/model_registry.py`'s `pick_winner()` actually selected for production (multi-task NN is never a candidate there - see the criterion note above).

| Task | Metric | Baseline (model) | Per-task-tuned XGBoost | Multi-task NN | Registered winner (model) |
|---|---|---|---|---|---|
| solubility | rmse | 0.890 (ridge_descriptors) | 0.709 | 0.660 | 0.699 (xgboost_tuned) |
| NR-AR | pr_auc | 0.523 (random_forest_descriptors) | 0.493 | 0.315 | 0.510 (random_forest_descriptors) |
| NR-AR-LBD | pr_auc | 0.645 (random_forest_descriptors) | 0.606 | 0.290 | 0.637 (random_forest_descriptors) |
| NR-AhR | pr_auc | 0.388 (logistic_regression_fingerprints) | 0.314 | 0.290 | 0.384 (logistic_regression_fingerprints) |
| NR-Aromatase | pr_auc | 0.174 (logistic_regression_fingerprints) | 0.111 | 0.097 | 0.132 (logistic_regression_fingerprints) |
| NR-ER | pr_auc | 0.334 (logistic_regression_fingerprints) | 0.308 | 0.274 | 0.258 (logistic_regression_fingerprints) |
| NR-ER-LBD | pr_auc | 0.311 (random_forest_descriptors) | 0.230 | 0.144 | 0.313 (random_forest_descriptors) |
| NR-PPAR-gamma | pr_auc | 0.127 (logistic_regression_fingerprints) | 0.102 | 0.024 | 0.115 (logistic_regression_fingerprints) |
| SR-ARE | pr_auc | 0.400 (random_forest_descriptors) | 0.352 | 0.307 | 0.399 (random_forest_descriptors) |
| SR-ATAD5 | pr_auc | 0.151 (random_forest_descriptors) | 0.123 | 0.045 | 0.151 (random_forest_descriptors) |
| SR-HSE | pr_auc | 0.180 (random_forest_descriptors) | 0.137 | 0.111 | 0.191 (random_forest_descriptors) |
| SR-MMP | pr_auc | 0.489 (random_forest_descriptors) | 0.513 | 0.440 | 0.534 (xgboost_tuned) |
| SR-p53 | pr_auc | 0.210 (logistic_regression_fingerprints) | 0.241 | 0.139 | 0.207 (xgboost_tuned) |
| fda_approved | pr_auc | 0.988 (logistic_regression_fingerprints) | 0.975 | 0.986 | 0.989 (logistic_regression_fingerprints) |
| ct_tox | pr_auc | 0.369 (logistic_regression_fingerprints) | 0.157 | 0.230 | 0.399 (logistic_regression_fingerprints) |
| bbbp_penetration | pr_auc | 0.905 (logistic_regression_fingerprints) | 0.876 | 0.818 | 0.913 (logistic_regression_fingerprints) |

## Artifact size and inference latency, per task

Latency = mean wall-clock time for one single-molecule prediction call, timeit-measured over 50 sampled rows x 20 repeats (a simple micro-benchmark, not a production load test). Artifact size = bytes of that task's model bundle alone, pickled in-memory (not the combined 125MB file).

| Task | Model | Artifact size | Latency (ms/prediction) |
|---|---|---|---|
| solubility | xgboost_tuned | 340.5 KB | 0.013 |
| NR-AR | random_forest_descriptors | 15625.5 KB | 2.191 |
| NR-AR-LBD | random_forest_descriptors | 12574.4 KB | 2.432 |
| NR-AhR | logistic_regression_fingerprints | 26.5 KB | 0.045 |
| NR-Aromatase | logistic_regression_fingerprints | 26.5 KB | 0.048 |
| NR-ER | logistic_regression_fingerprints | 26.5 KB | 0.054 |
| NR-ER-LBD | random_forest_descriptors | 18218.5 KB | 2.757 |
| NR-PPAR-gamma | logistic_regression_fingerprints | 26.5 KB | 0.054 |
| SR-ARE | random_forest_descriptors | 36512.5 KB | 2.814 |
| SR-ATAD5 | random_forest_descriptors | 16188.0 KB | 2.559 |
| SR-HSE | random_forest_descriptors | 20718.9 KB | 2.704 |
| SR-MMP | xgboost_tuned | 764.4 KB | 0.063 |
| SR-p53 | xgboost_tuned | 725.9 KB | 0.061 |
| fda_approved | logistic_regression_fingerprints | 26.5 KB | 0.042 |
| ct_tox | logistic_regression_fingerprints | 26.5 KB | 0.050 |
| bbbp_penetration | logistic_regression_fingerprints | 26.5 KB | 0.049 |
