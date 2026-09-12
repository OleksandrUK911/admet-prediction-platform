# Calibration + applicability domain report

## Expected Calibration Error (ECE) comparison

ECE measured on the val split (never used to fit or calibrate the model). Lower ECE = predicted probabilities better match true positive rates. ROC-AUC is included as a sanity check: calibration reshapes probabilities but should not change ranking, so ROC-AUC should stay roughly flat across the three methods.

| Task | Method | N (val) | ECE | ROC-AUC |
|---|---|---|---|---|
| NR-AR | raw_rf | 1660 | 0.012 | 0.808 |
| NR-AR | platt_sigmoid | 1660 | 0.016 | 0.822 |
| NR-AR | isotonic | 1660 | 0.013 | 0.812 |
| SR-MMP | raw_rf | 1378 | 0.080 | 0.783 |
| SR-MMP | platt_sigmoid | 1378 | 0.075 | 0.768 |
| SR-MMP | isotonic | 1378 | 0.079 | 0.771 |
| bbbp_penetration | raw_rf | 260 | 0.091 | 0.814 |
| bbbp_penetration | platt_sigmoid | 260 | 0.085 | 0.804 |
| bbbp_penetration | isotonic | 260 | 0.093 | 0.808 |
| ct_tox | raw_rf | 269 | 0.033 | 0.694 |
| ct_tox | platt_sigmoid | 269 | 0.015 | 0.727 |
| ct_tox | isotonic | 269 | 0.026 | 0.722 |

### Which method won, per task

- **NR-AR**: neither calibration method beat raw RF (raw ECE=0.012 was lowest)
- **SR-MMP**: platt_sigmoid won (ECE=0.075 vs. raw RF ECE=0.080)
- **bbbp_penetration**: platt_sigmoid won (ECE=0.085 vs. raw RF ECE=0.091)
- **ct_tox**: platt_sigmoid won (ECE=0.015 vs. raw RF ECE=0.033)

## Applicability domain (descriptor-space nearest-neighbor distance)

For each task's test-split molecules: Euclidean distance (in train-fit-scaled 7-descriptor space) to the nearest TRAIN molecule for that task. The top 9% by distance are flagged out-of-domain (OOD); raw-RF classification accuracy (threshold 0.5) is compared between the OOD group and the rest.

| Task | N test | N OOD | OOD mean NN dist | In-domain mean NN dist | OOD accuracy | In-domain accuracy |
|---|---|---|---|---|---|---|
| NR-AR | 598 | 60 | 1.514 | 0.310 | 0.950 | 0.959 |
| SR-MMP | 465 | 47 | 1.315 | 0.308 | 0.766 | 0.854 |
| bbbp_penetration | 166 | 17 | 1.581 | 0.402 | 0.647 | 0.872 |
| ct_tox | 129 | 13 | 1.540 | 0.347 | 0.923 | 0.922 |

### Did applicability domain predict worse accuracy?

- **NR-AR**: no meaningful difference - OOD accuracy 0.950 vs. in-domain 0.959 (gap 0.009).
- **SR-MMP**: yes - OOD accuracy 0.766 vs. in-domain 0.854 (gap 0.088).
- **bbbp_penetration**: yes - OOD accuracy 0.647 vs. in-domain 0.872 (gap 0.225).
- **ct_tox**: no meaningful difference - OOD accuracy 0.923 vs. in-domain 0.922 (gap -0.001).
