# Multitask NN vs. per-task-best comparison

One shared-backbone (7 -> 64 -> 64, ReLU) multi-task MLP with 16 task-specific linear heads (15 classification logits for BCE loss, 1 regression output for MSE loss), trained jointly on all 16 tasks. Missing labels are masked per-task per-row in the loss (`(loss * mask).sum() / mask.sum()`), never imputed. Per-task loss contributions are weighted by fixed inverse-frequency weights (1 / train-split non-missing count, renormalized to sum to 1) so Tox21's 12 tasks (~5700-7100 rows each) don't drown out ClinTox's 2 tasks (~1400 rows) or BBBP (~1949) or solubility (~1115).

"Per-task-best" below is whichever model - baseline (majority/logreg/random-forest) or per-task-tuned XGBoost - had the best val score for that task, from `ml/results/baseline_metrics.json` and `ml/results/per_task_xgboost_metrics.json`.

## Regression (solubility)

| Split | Per-task-best | Best RMSE | Multitask RMSE | Multitask R2 | Multitask wins? |
|---|---|---|---|---|---|
| val | xgboost_tuned | 0.709 | 0.660 | 0.804 | Yes |
| test | xgboost_tuned | 0.904 | 0.706 | 0.923 | Yes |

## Classification (15 tasks)

| Task | Split | Per-task-best | Best ROC-AUC | Best PR-AUC | Multitask ROC-AUC | Multitask PR-AUC | Multitask wins (val ROC-AUC)? |
|---|---|---|---|---|---|---|---|
| NR-AR | val | random_forest_descriptors | 0.808 | 0.523 | 0.813 | 0.315 | Yes |
| NR-AR | test | random_forest_descriptors | 0.610 | 0.208 | 0.722 | 0.207 | Yes |
| NR-AR-LBD | val | logistic_regression_fingerprints | 0.893 | 0.580 | 0.756 | 0.290 | No |
| NR-AR-LBD | test | logistic_regression_fingerprints | 0.785 | 0.461 | 0.713 | 0.072 | No |
| NR-AhR | val | logistic_regression_fingerprints | 0.768 | 0.388 | 0.731 | 0.290 | No |
| NR-AhR | test | logistic_regression_fingerprints | 0.841 | 0.476 | 0.775 | 0.431 | No |
| NR-Aromatase | val | xgboost_tuned | 0.768 | 0.111 | 0.755 | 0.097 | No |
| NR-Aromatase | test | xgboost_tuned | 0.751 | 0.196 | 0.646 | 0.149 | No |
| NR-ER | val | logistic_regression_fingerprints | 0.647 | 0.334 | 0.630 | 0.274 | No |
| NR-ER | test | logistic_regression_fingerprints | 0.697 | 0.347 | 0.655 | 0.252 | No |
| NR-ER-LBD | val | logistic_regression_fingerprints | 0.770 | 0.303 | 0.685 | 0.144 | No |
| NR-ER-LBD | test | logistic_regression_fingerprints | 0.790 | 0.289 | 0.755 | 0.129 | No |
| NR-PPAR-gamma | val | random_forest_descriptors | 0.783 | 0.126 | 0.495 | 0.024 | No |
| NR-PPAR-gamma | test | random_forest_descriptors | 0.792 | 0.183 | 0.714 | 0.100 | No |
| SR-ARE | val | random_forest_descriptors | 0.747 | 0.400 | 0.688 | 0.307 | No |
| SR-ARE | test | random_forest_descriptors | 0.758 | 0.468 | 0.633 | 0.285 | No |
| SR-ATAD5 | val | logistic_regression_fingerprints | 0.714 | 0.128 | 0.582 | 0.045 | No |
| SR-ATAD5 | test | logistic_regression_fingerprints | 0.740 | 0.088 | 0.783 | 0.189 | Yes |
| SR-HSE | val | random_forest_descriptors | 0.689 | 0.180 | 0.645 | 0.111 | No |
| SR-HSE | test | random_forest_descriptors | 0.692 | 0.115 | 0.683 | 0.116 | No |
| SR-MMP | val | random_forest_descriptors | 0.783 | 0.489 | 0.801 | 0.440 | Yes |
| SR-MMP | test | random_forest_descriptors | 0.828 | 0.586 | 0.747 | 0.416 | No |
| SR-p53 | val | xgboost_tuned | 0.739 | 0.241 | 0.687 | 0.139 | No |
| SR-p53 | test | xgboost_tuned | 0.764 | 0.296 | 0.753 | 0.226 | No |
| fda_approved | val | logistic_regression_fingerprints | 0.823 | 0.988 | 0.782 | 0.986 | No |
| fda_approved | test | logistic_regression_fingerprints | 0.807 | 0.974 | 0.690 | 0.955 | No |
| ct_tox | val | logistic_regression_fingerprints | 0.785 | 0.369 | 0.776 | 0.230 | No |
| ct_tox | test | logistic_regression_fingerprints | 0.762 | 0.273 | 0.738 | 0.181 | No |
| bbbp_penetration | val | random_forest_descriptors | 0.814 | 0.895 | 0.705 | 0.818 | No |
| bbbp_penetration | test | random_forest_descriptors | 0.811 | 0.896 | 0.858 | 0.934 | Yes |

**Summary**: on val, the multitask NN beat the per-task-best model on 3 of 16 tasks (regression: 1/1, classification: 2/15 with a defined ROC-AUC comparison). Examples where multitask helped: NR-AR, SR-MMP. Examples where it did not: NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD. Honest verdict: the result is mixed - multitask helps a minority of tasks, not a clear overall win. This is directionally consistent with the per-task XGBoost finding (only 3/15 classification tasks improved on baseline there) that, at this dataset size (~10k molecules total, most tasks missing labels for most rows), model choice matters less than the fundamental data scarcity per task.
