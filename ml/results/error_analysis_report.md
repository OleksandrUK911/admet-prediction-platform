# Error Analysis: False Positives / False Negatives (Toxicity + ADMET Classification Tasks)

Val-split predictions from the ACTUAL registered production models (`models/production/models.joblib`, reloaded the same way as `ml/evaluate.py` - not freshly retrained), at a plain 0.5 decision threshold. 15 of 15 classification tasks scored.

Two lenses: (1) Murcko scaffold groups (via `ml/preprocess.py`'s `murcko_scaffold`) with an error rate notably above the task's overall error rate (>= 1.5x, group size >= 5 molecules in val); (2) whether the 7 shared physico-chemical descriptors (`ml/features.py`'s `DESCRIPTOR_COLUMNS`) differ between error groups (FP/FN) and correctly-classified molecules by >= 0.5 standard deviations of the correct group's spread.

This is exploratory and reported honestly: several tasks show no scaffold group or descriptor pattern that clears these (fairly loose) thresholds - that is stated plainly below rather than stretched into a finding.

## Cross-task pattern worth flagging

Nearly every Tox21 task below is FN-heavy (false negatives far outnumber false positives, often 0 FP at all) - a direct consequence of these tasks' low positive prevalence plus a plain 0.5 threshold: the calibrated models learn to require strong evidence before predicting the rare positive class, so they miss real positives more often than they falsely flag negatives. This is a threshold/calibration artifact, not evidence the models ignore structure - see the caveats below.
The one recurring structural signal that shows up across several *different* nuclear receptor assays (NR-AR, NR-AR-LBD, NR-ER, NR-ER-LBD, and to a smaller extent bbbp_penetration) is the steroid-ketone scaffold `O=C1C=C2CCC3C4CCCC4CCC3C2CC1` (an androstenedione/steroid-hormone-like core): it is a small group in val (24-33 molecules) but its error rate is 6-24x the task's overall error rate, almost entirely false negatives. This makes biological sense - these four assays specifically test binding to steroid hormone receptors (androgen/estrogen), so real steroid-scaffold agonists/antagonists are exactly the molecules the model most needs to recognize, and it is instead systematically missing them. This is the one finding in this report that looks like a genuine, actionable structural blind spot rather than noise.
A second, weaker recurring signal: the benzanilide-like scaffold `O=C(Nc1ccccc1)c1ccccc1` and the stilbene-like scaffold `C(=Cc1ccccc1)c1ccccc1` reappear as over-represented FN groups across several unrelated SR-* stress-response assays (SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53) - each individually a small group (7-20 molecules), so treat this as suggestive rather than conclusive.
For fda_approved, ct_tox, and bbbp_penetration - the 3 non-Tox21 tasks with much smaller val sets (260-269 rows) - no scaffold group has enough repeated molecules to say anything structural; their descriptor shifts (e.g. higher LogP/RingCount in fda_approved's false positives) are the only signal available and are modest (z just above the 0.5 threshold).

## Summary table

| task | model | n (val) | FP | FN | error rate | scaffold groups considered | over-represented scaffolds |
|---|---|---|---|---|---|---|---|
| NR-AR | random_forest_descriptors | 1660 | 0 | 60 | 0.036 | 9 | 4 |
| NR-AR-LBD | random_forest_descriptors | 1553 | 1 | 43 | 0.028 | 8 | 1 |
| NR-AhR | logistic_regression_fingerprints | 1494 | 0 | 208 | 0.139 | 8 | 2 |
| NR-Aromatase | logistic_regression_fingerprints | 1329 | 0 | 45 | 0.034 | 7 | 2 |
| NR-ER | logistic_regression_fingerprints | 1436 | 0 | 200 | 0.139 | 9 | 5 |
| NR-ER-LBD | random_forest_descriptors | 1604 | 1 | 71 | 0.045 | 9 | 2 |
| NR-PPAR-gamma | logistic_regression_fingerprints | 1515 | 0 | 30 | 0.020 | 8 | 1 |
| SR-ARE | random_forest_descriptors | 1425 | 0 | 201 | 0.141 | 8 | 4 |
| SR-ATAD5 | random_forest_descriptors | 1631 | 0 | 48 | 0.029 | 9 | 3 |
| SR-HSE | random_forest_descriptors | 1565 | 0 | 77 | 0.049 | 10 | 3 |
| SR-MMP | xgboost_tuned | 1378 | 8 | 225 | 0.169 | 7 | 1 |
| SR-p53 | xgboost_tuned | 1583 | 0 | 86 | 0.054 | 9 | 4 |
| fda_approved | logistic_regression_fingerprints | 269 | 14 | 0 | 0.052 | 2 | 0 |
| ct_tox | logistic_regression_fingerprints | 269 | 0 | 17 | 0.063 | 2 | 0 |
| bbbp_penetration | logistic_regression_fingerprints | 260 | 55 | 3 | 0.223 | 3 | 2 |

## Per-task detail

### NR-AR (model: random_forest_descriptors)

n_val=1660, FP=0, FN=60, overall error rate=0.036, 9 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C1C=C2CCC3C4CCCC4CCC3C2CC1 | 32 | 0 | 26 | 0.812 | 22.5x |
| C(=Cc1ccccc1)c1ccccc1 | 8 | 0 | 1 | 0.125 | 3.5x |
| C1CNCCN1 | 9 | 0 | 1 | 0.111 | 3.1x |
| c1ccc2[nH]cnc2c1 | 12 | 0 | 1 | 0.083 | 2.3x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | MolWt | 338.35 | 235.26 | +0.78 |
| FN | LogP | 3.28 | 2.39 | +0.53 |
| FN | RingCount | 2.67 | 1.31 | +1.43 |

### NR-AR-LBD (model: random_forest_descriptors)

n_val=1553, FP=1, FN=43, overall error rate=0.028, 8 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C1C=C2CCC3C4CCCC4CCC3C2CC1 | 33 | 0 | 22 | 0.667 | 23.5x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | MolWt | 313.60 | 231.01 | +0.63 |
| FN | RingCount | 2.63 | 1.30 | +1.46 |

### NR-AhR (model: logistic_regression_fingerprints)

n_val=1494, FP=0, FN=208, overall error rate=0.139, 8 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C(Nc1ccccc1)c1ccccc1 | 19 | 0 | 13 | 0.684 | 4.9x |
| c1ccc2[nH]cnc2c1 | 10 | 0 | 6 | 0.600 | 4.3x |

**Descriptor analysis:** no descriptor differs from the correctly-classified group by >= 0.5 std devs for either FP or FN - errors do not obviously cluster in a particular size/LogP/polarity region for this task.

### NR-Aromatase (model: logistic_regression_fingerprints)

n_val=1329, FP=0, FN=45, overall error rate=0.034, 7 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1 | 6 | 0 | 1 | 0.167 | 4.9x |
| C1CNCCN1 | 8 | 0 | 1 | 0.125 | 3.7x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | MolWt | 295.62 | 229.21 | +0.51 |
| FN | LogP | 3.17 | 2.24 | +0.56 |
| FN | RingCount | 1.71 | 1.27 | +0.51 |

### NR-ER (model: logistic_regression_fingerprints)

n_val=1436, FP=0, FN=200, overall error rate=0.139, 9 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C1C=C2CCC3C4CCCC4CCC3C2CC1 | 28 | 0 | 23 | 0.821 | 5.9x |
| C(=Cc1ccccc1)c1ccccc1 | 7 | 0 | 5 | 0.714 | 5.1x |
| O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1 | 5 | 0 | 3 | 0.600 | 4.3x |
| O=C(Nc1ccccc1)c1ccccc1 | 10 | 0 | 4 | 0.400 | 2.9x |
| C1CCCC1 | 5 | 0 | 2 | 0.400 | 2.9x |

**Descriptor analysis:** no descriptor differs from the correctly-classified group by >= 0.5 std devs for either FP or FN - errors do not obviously cluster in a particular size/LogP/polarity region for this task.

### NR-ER-LBD (model: random_forest_descriptors)

n_val=1604, FP=1, FN=71, overall error rate=0.045, 9 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| C(=Cc1ccccc1)c1ccccc1 | 8 | 0 | 4 | 0.500 | 11.1x |
| O=C1C=C2CCC3C4CCCC4CCC3C2CC1 | 29 | 1 | 9 | 0.345 | 7.7x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | LogP | 3.20 | 2.34 | +0.52 |

### NR-PPAR-gamma (model: logistic_regression_fingerprints)

n_val=1515, FP=0, FN=30, overall error rate=0.020, 8 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C(Nc1ccccc1)c1ccccc1 | 13 | 0 | 1 | 0.077 | 3.9x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | MolWt | 321.42 | 228.49 | +0.78 |

### SR-ARE (model: random_forest_descriptors)

n_val=1425, FP=0, FN=201, overall error rate=0.141, 8 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| C(=Cc1ccccc1)c1ccccc1 | 7 | 0 | 4 | 0.571 | 4.1x |
| O=C(Nc1ccccc1)c1ccccc1 | 11 | 0 | 5 | 0.455 | 3.2x |
| c1ccc(N2CCNCC2)cc1 | 7 | 0 | 3 | 0.429 | 3.0x |
| O=C1C=C2CCC3C4CCCC4CCC3C2CC1 | 23 | 0 | 7 | 0.304 | 2.2x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | RingCount | 1.64 | 1.21 | +0.60 |

### SR-ATAD5 (model: random_forest_descriptors)

n_val=1631, FP=0, FN=48, overall error rate=0.029, 9 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| c1ccc2[nH]cnc2c1 | 12 | 0 | 3 | 0.250 | 8.5x |
| C(=Cc1ccccc1)c1ccccc1 | 7 | 0 | 1 | 0.143 | 4.9x |
| O=C(Nc1ccccc1)c1ccccc1 | 20 | 0 | 2 | 0.100 | 3.4x |

**Descriptor analysis:** no descriptor differs from the correctly-classified group by >= 0.5 std devs for either FP or FN - errors do not obviously cluster in a particular size/LogP/polarity region for this task.

### SR-HSE (model: random_forest_descriptors)

n_val=1565, FP=0, FN=77, overall error rate=0.049, 10 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C(Nc1ccccc1)c1ccccc1 | 16 | 0 | 4 | 0.250 | 5.1x |
| c1ncc2nc[nH]c2n1 | 5 | 0 | 1 | 0.200 | 4.1x |
| c1ccc2[nH]cnc2c1 | 11 | 0 | 1 | 0.091 | 1.8x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | LogP | 3.23 | 2.29 | +0.63 |

### SR-MMP (model: xgboost_tuned)

n_val=1378, FP=8, FN=225, overall error rate=0.169, 7 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C(Nc1ccccc1)c1ccccc1 | 15 | 0 | 7 | 0.467 | 2.8x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FP | MolWt | 349.38 | 228.44 | +0.94 |
| FP | LogP | 4.47 | 2.15 | +1.41 |
| FP | RingCount | 2.50 | 1.29 | +1.40 |
| FN | LogP | 3.06 | 2.15 | +0.55 |

### SR-p53 (model: xgboost_tuned)

n_val=1583, FP=0, FN=86, overall error rate=0.054, 9 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| c1ccc2[nH]cnc2c1 | 12 | 0 | 6 | 0.500 | 9.2x |
| O=C(Nc1ccccc1)c1ccccc1 | 17 | 0 | 5 | 0.294 | 5.4x |
| C(=Cc1ccccc1)c1ccccc1 | 7 | 0 | 2 | 0.286 | 5.3x |
| C1CCCC1 | 7 | 0 | 1 | 0.143 | 2.6x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | MolWt | 327.18 | 230.72 | +0.79 |
| FN | TPSA | 72.96 | 50.24 | +0.57 |
| FN | NumHAcceptors | 3.90 | 2.71 | +0.54 |
| FN | RingCount | 2.00 | 1.31 | +0.75 |

### fda_approved (model: logistic_regression_fingerprints)

n_val=269, FP=14, FN=0, overall error rate=0.052, 2 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** no scaffold group reaches the over-representation threshold (1.5x the task's overall error rate at n>=5) - errors here do not concentrate in any one structural scaffold class large enough to say anything about; they look spread across the val set.

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FP | LogP | 2.75 | 1.01 | +0.55 |
| FP | RingCount | 3.00 | 1.99 | +0.71 |

### ct_tox (model: logistic_regression_fingerprints)

n_val=269, FP=0, FN=17, overall error rate=0.063, 2 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** no scaffold group reaches the over-representation threshold (1.5x the task's overall error rate at n>=5) - errors here do not concentrate in any one structural scaffold class large enough to say anything about; they look spread across the val set.

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FN | LogP | 2.66 | 1.00 | +0.53 |
| FN | RingCount | 2.76 | 2.00 | +0.55 |

### bbbp_penetration (model: logistic_regression_fingerprints)

n_val=260, FP=55, FN=3, overall error rate=0.223, 3 scaffold group(s) with >= 5 molecules.

**Scaffold analysis:** over-represented scaffold group(s):

| scaffold (SMILES) | n in group | FP | FN | group error rate | vs task overall |
|---|---|---|---|---|---|
| O=C1C=C2CCC3C4CCCC4CCC3C2CC1 | 24 | 8 | 1 | 0.375 | 1.7x |
| O=C(Cc1ccccc1)NC1C(=O)N2C=CCSC12 | 8 | 3 | 0 | 0.375 | 1.7x |

**Descriptor analysis:** notable shift(s) (>= 0.5 std dev vs correctly-classified molecules):

| group | descriptor | group mean | correct-group mean | z |
|---|---|---|---|---|
| FP | NumHDonors | 2.20 | 1.53 | +0.50 |

## Caveats

- 0.5 is a default decision threshold, not a tuned operating point - FP/FN counts here would shift with a different threshold, and (per `TODO/ml/TODO_evaluation_validation.md`) false negatives and false positives likely have different real-world costs for a toxicity task that this analysis does not weigh.
- Many val sets have only a handful of scaffold groups with >= 5 molecules (most scaffolds in this dataset are singletons - see the summary table's "scaffold groups considered" column) - absence of a flagged scaffold group often means there just isn't enough repetition of any one scaffold to say something about it, not that scaffold has no effect.
- Descriptor z-scores are computed against the correctly-classified group's own spread per task, so they are not comparable in absolute terms across tasks.
