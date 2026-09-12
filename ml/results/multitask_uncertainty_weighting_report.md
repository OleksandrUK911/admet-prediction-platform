# Uncertainty weighting vs. fixed inverse-frequency weighting (multitask NN)

Same shared-backbone multi-task MLP (7 -> 64 -> 64, 16 task-specific heads), same data/split/masking as ml/experiments_multitask.py. The only difference: task weights are learned jointly with the network (Kendall et al.-style homoscedastic uncertainty weighting, `loss = sum_i 0.5 * exp(-2*log_sigma_i) * loss_i + log_sigma_i`, `log_sigma_i` a learnable per-task parameter starting at 0) instead of ml/experiments_multitask.py's fixed inverse-frequency weights computed once before training. Compares val metrics between the two weighting schemes, per task - **not** against per-task XGBoost/baseline models (see ml/results/multitask_comparison_report.md for that comparison; both multitask variants here are compared only against each other).

## Regression (solubility)

| Task | Split | Fixed-weighting RMSE | Uncertainty-weighting RMSE | Uncertainty wins? |
|---|---|---|---|---|
| solubility | val | 0.660 | 0.703 | No |

## Classification (15 tasks)

| Task | Fixed-weighting ROC-AUC | Fixed-weighting PR-AUC | Uncertainty-weighting ROC-AUC | Uncertainty-weighting PR-AUC | Uncertainty wins (val ROC-AUC)? |
|---|---|---|---|---|---|
| NR-AR | 0.813 | 0.315 | 0.760 | 0.427 | No |
| NR-AR-LBD | 0.756 | 0.290 | 0.856 | 0.434 | Yes |
| NR-AhR | 0.731 | 0.290 | 0.766 | 0.352 | Yes |
| NR-Aromatase | 0.755 | 0.097 | 0.777 | 0.134 | Yes |
| NR-ER | 0.630 | 0.274 | 0.609 | 0.260 | No |
| NR-ER-LBD | 0.685 | 0.144 | 0.709 | 0.216 | Yes |
| NR-PPAR-gamma | 0.495 | 0.024 | 0.688 | 0.051 | Yes |
| SR-ARE | 0.688 | 0.307 | 0.739 | 0.377 | Yes |
| SR-ATAD5 | 0.582 | 0.045 | 0.624 | 0.081 | Yes |
| SR-HSE | 0.645 | 0.111 | 0.673 | 0.133 | Yes |
| SR-MMP | 0.801 | 0.440 | 0.819 | 0.521 | Yes |
| SR-p53 | 0.687 | 0.139 | 0.732 | 0.199 | Yes |
| fda_approved | 0.782 | 0.986 | 0.776 | 0.985 | No |
| ct_tox | 0.776 | 0.230 | 0.774 | 0.244 | No |
| bbbp_penetration | 0.705 | 0.818 | 0.717 | 0.835 | Yes |

**Summary**: on val, uncertainty weighting beat fixed inverse-frequency weighting on 11/16 tasks (regression: 0/1, classification: 11/15). Examples where uncertainty weighting helped: NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER-LBD, NR-PPAR-gamma. Examples where it did not: NR-AR, NR-ER, fda_approved, ct_tox. Honest verdict: uncertainty weighting beat fixed inverse-frequency weighting on most tasks. This project has repeatedly found the multitask NN underperforms per-task models overall (3/16 wins vs. per-task-best - see ml/results/multitask_comparison_report.md); swapping the weighting scheme changes which/how many tasks the multitask NN wins **against its own fixed-weighting variant**, but is not expected to and does not, on its own, close the larger gap against per-task models - the fundamental constraint remains data scarcity per task on this ~10k-molecule dataset, not the choice of task weighting scheme.
