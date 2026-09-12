# SMOTE Oversampling Experiment (Most-Imbalanced Classification Tasks)

The 4 classification tasks with the lowest actual train-split positive rate (recomputed directly from `data/processed/admet_processed.csv`, not assumed) get a SMOTE-oversampled training run, using the SAME classifier architecture as that task's current production winner (`models/production/metadata.json` -> `ml/model_registry.py`'s `build_classifier`), on Morgan fingerprint features. Both the plain-train and SMOTE-oversampled model are scored on the untouched val split by PR-AUC - the metric this project uses for model selection throughout.

This is exploratory only - it does NOT change what is registered in `models/production/models.joblib`, regardless of outcome.

## Results

| task | model | train n (pos/neg) | pos rate | val PR-AUC (no SMOTE) | val PR-AUC (SMOTE) | delta | helped? |
|---|---|---|---|---|---|---|---|
| NR-PPAR-gamma | logistic_regression_fingerprints | 135/4177 | 3.1% | 0.127 | 0.091 | -0.035 | No (worse) |
| NR-AR-LBD | random_forest_descriptors | 165/4374 | 3.6% | 0.660 | 0.679 | +0.019 | Yes |
| SR-ATAD5 | random_forest_descriptors | 192/4556 | 4.0% | 0.174 | 0.167 | -0.007 | No (worse) |
| NR-AR | random_forest_descriptors | 207/4671 | 4.2% | 0.549 | 0.536 | -0.013 | No (worse) |

## Honest result

Of 4 task(s) where SMOTE could run, it improved val PR-AUC (by more than a 0.005 margin) on 1, made it measurably worse on 3, and left it essentially unchanged on 0. 0 task(s) were skipped (too few minority-class training examples for SMOTE's neighbor search).

**Mixed result**: SMOTE helped on 1/4 scored tasks and did not on the rest - reported as-is rather than generalized into a blanket recommendation either way.

## Caveats

- SMOTE is applied on 1024-bit Morgan fingerprint features - synthetic neighbors are interpolated bit vectors that do not correspond to real molecules; this is standard practice for SMOTE but worth stating plainly since "chemical validity" of the synthesized examples is not checked.
- Val split and model architecture are identical to production - only the training-set class balance changes, so any effect observed is attributable to oversampling, not a confound from a different model or split.
- This experiment is not wired into ml/model_registry.py and does not change what ships.
- Both rows (no-SMOTE and SMOTE) here use fingerprint features for every task, including tasks whose actual production winner is `random_forest_descriptors` (i.e. normally trained on the 7 physico-chemical descriptors, not fingerprints) - this is deliberate (SMOTE needs one consistent feature space to compare fairly), but it means the "no-SMOTE" PR-AUC in this report can differ from that task's real `models/production/metadata.json` val PR-AUC, which may use descriptors instead. The comparison that matters here is no-SMOTE vs SMOTE, both on the same fingerprint features - not this report's no-SMOTE column vs production metadata.
