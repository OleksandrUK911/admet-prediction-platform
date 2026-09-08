# ADMET Prediction Platform

Multi-task ML platform predicting Absorption, Distribution, Metabolism,
Excretion and Toxicity (ADMET) properties from a molecule's SMILES, with
calibrated probabilities and uncertainty estimation.

## Status
🔴 In planning — implementation not started yet.

## Stack
- **ML:** Python, RDKit, scikit-learn/XGBoost, calibration (Platt/isotonic)
- **Backend:** FastAPI, PostgreSQL
- **Frontend:** React
- **Infra:** Docker, GitHub Actions

## How it works
```
SMILES → RDKit descriptors → multi-task ML → ADMET profile + confidence
```

## Disclaimer
Research/educational project only. **Not for clinical use.** Predictions
include explicit confidence/applicability-domain indicators and must not be
interpreted as medical or regulatory guidance.

## License
TBD
