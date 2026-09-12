# Descriptor relevance and multicollinearity report

## Descriptor-vs-label correlation, per task

Point-biserial correlation (= Pearson correlation with a binary variable) between each of the 7 descriptors and each task's label, computed on task_rows(df, task) - i.e. only rows with a non-missing label for that task. Top descriptor(s) are the ones with the largest |r|.

| Task | N | Top descriptor | r | 2nd descriptor | r |
|---|---|---|---|---|---|
| solubility | 1115 | LogP | -0.828 | MolWt | -0.639 |
| NR-AR | 7136 | RingCount | 0.195 | MolWt | 0.114 |
| NR-AR-LBD | 6634 | RingCount | 0.184 | MolWt | 0.104 |
| NR-AhR | 6439 | RingCount | 0.144 | LogP | 0.109 |
| NR-Aromatase | 5720 | LogP | 0.176 | RingCount | 0.150 |
| NR-ER | 6085 | RingCount | 0.143 | LogP | 0.104 |
| NR-ER-LBD | 6831 | RingCount | 0.106 | LogP | 0.098 |
| NR-PPAR-gamma | 6344 | LogP | 0.096 | MolWt | 0.086 |
| SR-ARE | 5749 | RingCount | 0.189 | MolWt | 0.173 |
| SR-ATAD5 | 6945 | RingCount | 0.077 | NumHAcceptors | 0.050 |
| SR-HSE | 6376 | LogP | 0.149 | MolWt | 0.091 |
| SR-MMP | 5714 | LogP | 0.295 | RingCount | 0.206 |
| SR-p53 | 6654 | RingCount | 0.179 | MolWt | 0.154 |
| fda_approved | 1427 | RingCount | -0.118 | LogP | -0.092 |
| ct_tox | 1427 | RingCount | 0.112 | LogP | 0.095 |
| bbbp_penetration | 1949 | TPSA | -0.536 | NumHDonors | -0.495 |

## Intuition check

- BBBP penetration: planning-doc intuition was TPSA/LogP should matter most. Top descriptor found: **TPSA** (TPSA r=-0.536, LogP r=0.311). Matches intuition.
- Solubility: planning-doc intuition was LogP/MolWt should matter most. Top descriptor found: **LogP** (LogP r=-0.828, MolWt r=-0.639). Matches intuition.
- Tox21 assays / ClinTox: no prior intuition was assumed - correlations reported as-found above, without forcing a narrative.

## Descriptor-descriptor multicollinearity (all ~10130 molecules)

Project #1 found (on ESOL, 1128 molecules) that TPSA correlates strongly with NumHDonors (r=0.755) and NumHAcceptors (r=0.899). Recomputed here on this project's larger, more chemically diverse ~10130-molecule set:

- TPSA vs NumHDonors: r = 0.847
- TPSA vs NumHAcceptors: r = 0.910

**Confirmed**: the TPSA/H-bonding multicollinearity holds on this larger, more diverse dataset too - consistent with it being a property of the descriptors' formulas (TPSA is computed as a sum over polar-atom surface-area contributions that are closely related to H-bond donor/acceptor counts) rather than an artifact specific to ESOL.

### Full descriptor-descriptor correlation matrix

| | MolWt | LogP | TPSA | NumHDonors | NumHAcceptors | NumRotatableBonds | RingCount |
|---|---|---|---|---|---|---|---|
| MolWt | 1.000 | 0.138 | 0.737 | 0.586 | 0.767 | 0.645 | 0.666 |
| LogP | 0.138 | 1.000 | -0.453 | -0.473 | -0.328 | 0.150 | 0.169 |
| TPSA | 0.737 | -0.453 | 1.000 | 0.847 | 0.910 | 0.496 | 0.420 |
| NumHDonors | 0.586 | -0.473 | 0.847 | 1.000 | 0.685 | 0.384 | 0.337 |
| NumHAcceptors | 0.767 | -0.328 | 0.910 | 0.685 | 1.000 | 0.506 | 0.505 |
| NumRotatableBonds | 0.645 | 0.150 | 0.496 | 0.384 | 0.506 | 1.000 | 0.116 |
| RingCount | 0.666 | 0.169 | 0.420 | 0.337 | 0.505 | 0.116 | 1.000 |

### Other notable descriptor pairs (|r| > 0.5, excluding the diagonal)

| Descriptor A | Descriptor B | r |
|---|---|---|
| TPSA | NumHAcceptors | 0.910 |
| TPSA | NumHDonors | 0.847 |
| MolWt | NumHAcceptors | 0.767 |
| MolWt | TPSA | 0.737 |
| NumHDonors | NumHAcceptors | 0.685 |
| MolWt | RingCount | 0.666 |
| MolWt | NumRotatableBonds | 0.645 |
| MolWt | NumHDonors | 0.586 |
| NumHAcceptors | NumRotatableBonds | 0.506 |
| NumHAcceptors | RingCount | 0.505 |
