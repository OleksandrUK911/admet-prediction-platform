# Per-task XGBoost vs. baseline comparison

One XGBoost model per task, trained on the 7 shared descriptors. Classification tasks use `scale_pos_weight` (train-split negatives / positives) to address class imbalance, instead of library defaults. Baseline-best is whichever of majority/logistic-regression/random-forest had the best val ROC-AUC (regression: best val RMSE) in `ml/results/baseline_metrics.json`.

Imbalanced tasks are judged primarily by PR-AUC, not ROC-AUC: ROC-AUC's false-positive-rate axis is diluted by the large number of true negatives typical of these tasks, so a classifier can look good on ROC-AUC while its positive predictions are still mostly wrong. PR-AUC, which only considers precision/recall over the (rare) positive class, is the more honest signal for whether class-weighting actually helped.

## Regression (solubility)

| Split | Baseline-best | Baseline RMSE | XGBoost RMSE | XGBoost R2 | Improved? |
|---|---|---|---|---|---|
| val | ridge_descriptors | 0.890 | 0.709 | 0.774 | Yes |
| test | ridge_descriptors | 0.876 | 0.904 | 0.873 | No |

## Classification (15 tasks)

| Task | Split | Baseline-best | Baseline ROC-AUC | Baseline PR-AUC | XGBoost ROC-AUC | XGBoost PR-AUC | Improved (val PR-AUC)? |
|---|---|---|---|---|---|---|---|
| NR-AR | val | random_forest_descriptors | 0.808 | 0.523 | 0.761 | 0.493 | No |
| NR-AR | test | random_forest_descriptors | 0.610 | 0.208 | 0.707 | 0.174 | No |
| NR-AR-LBD | val | logistic_regression_fingerprints | 0.893 | 0.580 | 0.860 | 0.606 | Yes |
| NR-AR-LBD | test | logistic_regression_fingerprints | 0.785 | 0.461 | 0.830 | 0.395 | No |
| NR-AhR | val | logistic_regression_fingerprints | 0.768 | 0.388 | 0.697 | 0.314 | No |
| NR-AhR | test | logistic_regression_fingerprints | 0.841 | 0.476 | 0.754 | 0.458 | No |
| NR-Aromatase | val | random_forest_descriptors | 0.755 | 0.125 | 0.768 | 0.111 | No |
| NR-Aromatase | test | random_forest_descriptors | 0.774 | 0.245 | 0.751 | 0.196 | No |
| NR-ER | val | logistic_regression_fingerprints | 0.647 | 0.334 | 0.622 | 0.308 | No |
| NR-ER | test | logistic_regression_fingerprints | 0.697 | 0.347 | 0.656 | 0.217 | No |
| NR-ER-LBD | val | logistic_regression_fingerprints | 0.770 | 0.303 | 0.724 | 0.230 | No |
| NR-ER-LBD | test | logistic_regression_fingerprints | 0.790 | 0.289 | 0.711 | 0.194 | No |
| NR-PPAR-gamma | val | random_forest_descriptors | 0.783 | 0.126 | 0.680 | 0.102 | No |
| NR-PPAR-gamma | test | random_forest_descriptors | 0.792 | 0.183 | 0.822 | 0.207 | Yes |
| SR-ARE | val | random_forest_descriptors | 0.747 | 0.400 | 0.716 | 0.352 | No |
| SR-ARE | test | random_forest_descriptors | 0.758 | 0.468 | 0.700 | 0.374 | No |
| SR-ATAD5 | val | logistic_regression_fingerprints | 0.714 | 0.128 | 0.672 | 0.123 | No |
| SR-ATAD5 | test | logistic_regression_fingerprints | 0.740 | 0.088 | 0.683 | 0.141 | Yes |
| SR-HSE | val | random_forest_descriptors | 0.689 | 0.180 | 0.626 | 0.137 | No |
| SR-HSE | test | random_forest_descriptors | 0.692 | 0.115 | 0.576 | 0.095 | No |
| SR-MMP | val | random_forest_descriptors | 0.783 | 0.489 | 0.776 | 0.513 | Yes |
| SR-MMP | test | random_forest_descriptors | 0.828 | 0.586 | 0.758 | 0.480 | No |
| SR-p53 | val | logistic_regression_fingerprints | 0.731 | 0.210 | 0.739 | 0.241 | Yes |
| SR-p53 | test | logistic_regression_fingerprints | 0.703 | 0.246 | 0.764 | 0.296 | Yes |
| fda_approved | val | logistic_regression_fingerprints | 0.823 | 0.988 | 0.713 | 0.975 | No |
| fda_approved | test | logistic_regression_fingerprints | 0.807 | 0.974 | 0.572 | 0.921 | No |
| ct_tox | val | logistic_regression_fingerprints | 0.785 | 0.369 | 0.722 | 0.157 | No |
| ct_tox | test | logistic_regression_fingerprints | 0.762 | 0.273 | 0.698 | 0.191 | No |
| bbbp_penetration | val | random_forest_descriptors | 0.814 | 0.895 | 0.777 | 0.876 | No |
| bbbp_penetration | test | random_forest_descriptors | 0.811 | 0.896 | 0.850 | 0.940 | Yes |

**Summary**: on val PR-AUC, XGBoost with `scale_pos_weight` improved on the baseline-best in 3 of 15 classification tasks with a defined comparison, and did not improve in 12.
