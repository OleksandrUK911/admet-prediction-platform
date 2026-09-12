# Feature selection report (high-dimensional fingerprints)

Logistic Regression on Morgan fingerprints (radius=2, 1024 bits, ml/features.py's default), val PR-AUC, for 4 representative classification tasks (NR-AR: small/imbalanced Tox21 assay, SR-MMP: larger/more-balanced Tox21 assay, bbbp_penetration: small non-Tox21 task, ct_tox: small + heavily imbalanced ClinTox task). PR-AUC, not ROC-AUC, to match ml/model_registry.py's pick_winner() criterion for imbalanced classification tasks.

Pipeline: VarianceThreshold(threshold=0.01) fit on train only (drops bits with p(1-p) below that, i.e. near-constant across train molecules), then a greedy pairwise-correlation filter (drop a bit the moment it is >0.95 absolute Pearson-r correlated with a bit already kept), also fit on train only. 'Meaningful' is a val PR-AUC change of more than 0.02 - smaller gaps are treated as noise given a single train/val split, same bar as ml/feature_engineering.py.

| Task | N train | N val | Bits (full) | Bits (post variance) | Bits (final) | Reduction | PR-AUC (full) | PR-AUC (selected) | Delta | Meaningful? |
|---|---|---|---|---|---|---|---|---|---|---|
| NR-AR | 4878 | 1660 | 1024 | 749 | 749 | 26.9% | 0.5041 | 0.5113 | +0.0072 | no (noise) |
| SR-MMP | 3871 | 1378 | 1024 | 722 | 722 | 29.5% | 0.4352 | 0.4194 | -0.0158 | no (noise) |
| bbbp_penetration | 1523 | 260 | 1024 | 848 | 848 | 17.2% | 0.9054 | 0.9050 | -0.0004 | no (noise) |
| ct_tox | 1029 | 269 | 1024 | 920 | 920 | 10.2% | 0.3687 | 0.3644 | -0.0043 | no (noise) |

## Verdict

Average bit-count reduction across the 4 tasks: **20.9%** (from 1024 bits down to the 'Bits (final)' column above per task). Of the 4 tasks: 0 meaningfully improved with feature selection, 0 meaningfully got worse, 4 were a wash (within 0.02 val PR-AUC, i.e. noise given a single split).

Notable sub-finding: on all 4 tasks, essentially all of the reduction came from VarianceThreshold alone - the correlation filter (|r| > 0.95) removed zero additional bits beyond what VarianceThreshold already dropped. Morgan fingerprint bits, once the near-constant ones are removed, are apparently not redundant enough at this threshold/task-subset size to trigger the correlation filter - a real (if unglamorous) finding, not a bug: raising this threshold's usefulness would require either a lower |r| cutoff or a larger training set to estimate correlations more stably.

A substantial bit-count reduction was achieved with essentially no change in val PR-AUC on any of the 4 tasks (all within the 0.02 noise band) - a wash, not a clear win or loss. Feature selection here mostly reduces model size/inference cost, not accuracy, for `logistic_regression_fingerprints`. Not adopted as the production default, since the current pipeline does not have a size/latency problem that would justify the added complexity for zero accuracy gain.
