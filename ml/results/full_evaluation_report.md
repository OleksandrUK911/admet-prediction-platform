# Full Production-Model Evaluation Report

## 1. Full production-model evaluation (train/val/test, reloaded bundles)

Metrics recomputed by loading the actual registered bundle from `models/production/models.joblib` for each of the 16 tasks (not a freshly-retrained copy) and scoring it on train/val/test.

| task | model | n_train | n_val | n_test | train metric | val metric | test metric |
|---|---|---|---|---|---|---|---|
| solubility | xgboost_tuned | 717 | 294 | 104 | RMSE=0.465 | RMSE=0.699 | RMSE=0.908 |
| NR-AR | random_forest_descriptors | 4878 | 1660 | 598 | PR-AUC=0.991 | PR-AUC=0.510 | PR-AUC=0.234 |
| NR-AR-LBD | random_forest_descriptors | 4539 | 1553 | 542 | PR-AUC=1.000 | PR-AUC=0.637 | PR-AUC=0.301 |
| NR-AhR | logistic_regression_fingerprints | 4425 | 1494 | 520 | PR-AUC=0.770 | PR-AUC=0.384 | PR-AUC=0.523 |
| NR-Aromatase | logistic_regression_fingerprints | 3902 | 1329 | 489 | PR-AUC=0.898 | PR-AUC=0.132 | PR-AUC=0.206 |
| NR-ER | logistic_regression_fingerprints | 4153 | 1436 | 496 | PR-AUC=0.663 | PR-AUC=0.258 | PR-AUC=0.365 |
| NR-ER-LBD | random_forest_descriptors | 4670 | 1604 | 557 | PR-AUC=0.986 | PR-AUC=0.313 | PR-AUC=0.244 |
| NR-PPAR-gamma | logistic_regression_fingerprints | 4312 | 1515 | 517 | PR-AUC=0.826 | PR-AUC=0.115 | PR-AUC=0.377 |
| SR-ARE | random_forest_descriptors | 3851 | 1425 | 473 | PR-AUC=0.995 | PR-AUC=0.399 | PR-AUC=0.426 |
| SR-ATAD5 | random_forest_descriptors | 4748 | 1631 | 566 | PR-AUC=0.999 | PR-AUC=0.151 | PR-AUC=0.185 |
| SR-HSE | random_forest_descriptors | 4292 | 1565 | 519 | PR-AUC=0.993 | PR-AUC=0.191 | PR-AUC=0.122 |
| SR-MMP | xgboost_tuned | 3871 | 1378 | 465 | PR-AUC=0.996 | PR-AUC=0.534 | PR-AUC=0.495 |
| SR-p53 | xgboost_tuned | 4535 | 1583 | 536 | PR-AUC=0.977 | PR-AUC=0.207 | PR-AUC=0.256 |
| fda_approved | logistic_regression_fingerprints | 1029 | 269 | 129 | PR-AUC=1.000 | PR-AUC=0.989 | PR-AUC=0.972 |
| ct_tox | logistic_regression_fingerprints | 1029 | 269 | 129 | PR-AUC=0.986 | PR-AUC=0.399 | PR-AUC=0.259 |
| bbbp_penetration | logistic_regression_fingerprints | 1523 | 260 | 166 | PR-AUC=0.999 | PR-AUC=0.913 | PR-AUC=0.938 |

### Consistency check vs `models/production/metadata.json`

All reloaded val/test metrics match the stored `per_task_winners` metadata (within 1e-4 relative tolerance).

### Train-vs-val overfitting gap check

Flagged tasks (train performance dramatically better than val - overfitting risk):

- **NR-AR**: train PR-AUC=0.991 vs val PR-AUC=0.510 (gap=0.481)
- **NR-AR-LBD**: train PR-AUC=1.000 vs val PR-AUC=0.637 (gap=0.363)
- **NR-AhR**: train PR-AUC=0.770 vs val PR-AUC=0.384 (gap=0.386)
- **NR-Aromatase**: train PR-AUC=0.898 vs val PR-AUC=0.132 (gap=0.766)
- **NR-ER**: train PR-AUC=0.663 vs val PR-AUC=0.258 (gap=0.405)
- **NR-ER-LBD**: train PR-AUC=0.986 vs val PR-AUC=0.313 (gap=0.673)
- **NR-PPAR-gamma**: train PR-AUC=0.826 vs val PR-AUC=0.115 (gap=0.711)
- **SR-ARE**: train PR-AUC=0.995 vs val PR-AUC=0.399 (gap=0.597)
- **SR-ATAD5**: train PR-AUC=0.999 vs val PR-AUC=0.151 (gap=0.848)
- **SR-HSE**: train PR-AUC=0.993 vs val PR-AUC=0.191 (gap=0.802)
- **SR-MMP**: train PR-AUC=0.996 vs val PR-AUC=0.534 (gap=0.462)
- **SR-p53**: train PR-AUC=0.977 vs val PR-AUC=0.207 (gap=0.770)
- **ct_tox**: train PR-AUC=0.986 vs val PR-AUC=0.399 (gap=0.587)

## 2. Cross-validation stability check (5-fold, train+val pool only)

Test split never touched. Each task's winning model is reconstructed via `ml/model_registry.py`'s `build_classifier`/`retrain_regressor` (same architecture and hyperparameters as production, but NOT wrapped in `CalibratedClassifierCV` - that step only affects probability calibration, not ranking metrics like ROC-AUC/PR-AUC, so it is skipped here to isolate the base model's stability).

| task | metric | mean | std | n_folds |
|---|---|---|---|---|
| solubility | mae | 0.5230 | 0.0351 | 5 |
| solubility | r2 | 0.8784 | 0.0073 | 5 |
| solubility | rmse | 0.6917 | 0.0468 | 5 |
| SR-MMP | pr_auc | 0.6626 | 0.0379 | 5 |
| SR-MMP | roc_auc | 0.8796 | 0.0120 | 5 |
| bbbp_penetration | pr_auc | 0.9612 | 0.0060 | 5 |
| bbbp_penetration | roc_auc | 0.9035 | 0.0088 | 5 |

Interpretation: a std that is small relative to the gap between candidate models (or relative to the single val-split number quoted in `models/production/metadata.json`) suggests the reported val metric is a stable estimate; a std comparable to or larger than that gap suggests the single train/val/test split could have been a lucky or unlucky draw for that task.

## 3. Whole-portfolio ranking

### Classification tasks, ranked by test PR-AUC (descending)

| rank | task | test PR-AUC | test ROC-AUC | n_positive/n_total |
|---|---|---|---|---|
| 1 | fda_approved | 0.972 | 0.793 | 1338/1427 |
| 2 | bbbp_penetration | 0.938 | 0.856 | 1478/1949 |
| 3 | NR-AhR | 0.523 | 0.860 | 762/6439 |
| 4 | SR-MMP | 0.495 | 0.783 | 903/5714 |
| 5 | SR-ARE | 0.426 | 0.763 | 935/5749 |
| 6 | NR-PPAR-gamma | 0.377 | 0.815 | 184/6344 |
| 7 | NR-ER | 0.365 | 0.698 | 768/6085 |
| 8 | NR-AR-LBD | 0.301 | 0.840 | 230/6634 |
| 9 | ct_tox | 0.259 | 0.769 | 84/1427 |
| 10 | SR-p53 | 0.256 | 0.764 | 419/6654 |
| 11 | NR-ER-LBD | 0.244 | 0.751 | 338/6831 |
| 12 | NR-AR | 0.234 | 0.698 | 299/7136 |
| 13 | NR-Aromatase | 0.206 | 0.619 | 295/5720 |
| 14 | SR-ATAD5 | 0.185 | 0.741 | 262/6945 |
| 15 | SR-HSE | 0.122 | 0.685 | 368/6376 |

### Regression task, by test R2

- **solubility**: test R2 = 0.872, test RMSE = 0.908

### 3 worst classification tasks - diagnosis

- **SR-HSE** (test PR-AUC=0.122, n_positive/n_total=368/6376 = 5.8%): test PR-AUC (0.122) close to the task's positive prevalence (5.8%) - near-baseline performance
- **SR-ATAD5** (test PR-AUC=0.185, n_positive/n_total=262/6945 = 3.8%): very few positives (262/6945 = 3.8% prevalence) - PR-AUC is hard to beat the low base rate with this little signal
- **NR-Aromatase** (test PR-AUC=0.206, n_positive/n_total=295/5720 = 5.2%): test PR-AUC (0.206) close to the task's positive prevalence (5.2%) - near-baseline performance
