# Fingerprint radius/bit-size comparison

Logistic Regression on Morgan fingerprints, val ROC-AUC, for 3 representative tasks (NR-AR: small/imbalanced Tox21 assay, SR-MMP: larger/more-balanced Tox21 assay, bbbp_penetration: small non-Tox21 task). 'Meaningful' is defined as a val ROC-AUC improvement of more than 0.02 over the current radius=2/1024-bit baseline used in ml/features.py - smaller gaps are treated as noise given val set sizes here.

| Task | Radius | Bits | N train | N val | Val ROC-AUC |
|---|---|---|---|---|---|
| NR-AR | 2 | 1024 | 4878 | 1660 | 0.7281 (current default) |
| NR-AR | 3 | 1024 | 4878 | 1660 | 0.7288 |
| NR-AR | 2 | 2048 | 4878 | 1660 | 0.7044 |
| NR-AR | 3 | 2048 | 4878 | 1660 | 0.7187 |
| SR-MMP | 2 | 1024 | 3871 | 1378 | 0.7662 (current default) |
| SR-MMP | 3 | 1024 | 3871 | 1378 | 0.7566 |
| SR-MMP | 2 | 2048 | 3871 | 1378 | 0.8003 |
| SR-MMP | 3 | 2048 | 3871 | 1378 | 0.7877 |
| bbbp_penetration | 2 | 1024 | 1523 | 260 | 0.8013 (current default) |
| bbbp_penetration | 3 | 1024 | 1523 | 260 | 0.7896 |
| bbbp_penetration | 2 | 2048 | 1523 | 260 | 0.8418 |
| bbbp_penetration | 3 | 2048 | 1523 | 260 | 0.8239 |

## Deltas vs. radius=2/1024 baseline

| Task | Variant | Val ROC-AUC | Delta vs baseline | Meaningful? |
|---|---|---|---|---|
| NR-AR | r=3/1024b | 0.7288 | +0.0007 | no (noise) |
| NR-AR | r=2/2048b | 0.7044 | -0.0237 | YES |
| NR-AR | r=3/2048b | 0.7187 | -0.0095 | no (noise) |
| SR-MMP | r=3/1024b | 0.7566 | -0.0095 | no (noise) |
| SR-MMP | r=2/2048b | 0.8003 | +0.0341 | YES |
| SR-MMP | r=3/2048b | 0.7877 | +0.0215 | YES |
| bbbp_penetration | r=3/1024b | 0.7896 | -0.0117 | no (noise) |
| bbbp_penetration | r=2/2048b | 0.8418 | +0.0405 | YES |
| bbbp_penetration | r=3/2048b | 0.8239 | +0.0226 | YES |

## Verdict

At least one variant crossed the 0.02 meaningful-delta threshold on at least one representative task - see the table above for which one(s). Given this is a single train/val split (not cross-validated), treat a single crossing with some caution before committing to a new default.
