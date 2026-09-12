"""Shared feature computation for ADMET baselines/experiments - one place
so every model-training script uses the exact same descriptor/fingerprint
definitions (per ml/TODO_baseline.md's "shared train/predict contract").

Reuses the same 7 physico-chemical descriptors as project #1 (portfolio
consistency) plus Morgan fingerprints (radius=2, 1024 bits, same as
project #1's choice).
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
}
DESCRIPTOR_COLUMNS = list(DESCRIPTOR_FUNCS.keys())

TOX21_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
CLASSIFICATION_TASKS = [*TOX21_ASSAYS, "fda_approved", "ct_tox", "bbbp_penetration"]
REGRESSION_TASKS = ["solubility"]
ALL_TASKS = REGRESSION_TASKS + CLASSIFICATION_TASKS


def compute_descriptors(smiles_series: pd.Series) -> pd.DataFrame:
    records = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        records.append({name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()})
    return pd.DataFrame(records)


def compute_fingerprints(smiles_series: pd.Series, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    fps = np.zeros((len(smiles_series), n_bits), dtype=np.uint8)
    for i, smiles in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(smiles)
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps[i] = np.array(bitvect)
    return fps


def task_rows(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Rows with a non-missing label for this task - masking, not imputation,
    per data/TODO_preprocessing_pipeline.md."""
    return df[df[task].notna()]
