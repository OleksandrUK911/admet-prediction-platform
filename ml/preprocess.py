"""Reproducible raw -> processed pipeline for the ADMET multi-task dataset.

Usage:
    python ml/preprocess.py

Self-downloads 4 MoleculeNet datasets (ESOL, Tox21, ClinTox, BBBP - see
data/README.md), canonicalizes SMILES with one consistent RDKit algorithm,
merges them into one multi-task table by canonical SMILES (missing labels
masked as NaN, not imputed - most compounds only appear in one source
dataset), computes 7 RDKit descriptors (same set as project #1, for
portfolio consistency), scaffold-splits train/val/test, and writes
data/processed/admet_processed.csv + data/processed/report.md.
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test

RAW_SOURCES = {
    "esol": ("delaney-processed.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"),
    "tox21": ("tox21.csv.gz", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"),
    "clintox": ("clintox.csv.gz", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"),
    "bbbp": ("BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"),
}

TOX21_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
TASK_COLUMNS = ["solubility", *TOX21_ASSAYS, "fda_approved", "ct_tox", "bbbp_penetration"]

DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
}

SALT_REMOVER = SaltRemover.SaltRemover()


def ensure_raw_datasets() -> None:
    """data/raw/ is gitignored - fetch every source on first run (locally
    or in CI) rather than requiring a manual download step."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (filename, url) in RAW_SOURCES.items():
        path = RAW_DIR / filename
        if path.exists():
            continue
        print(f"{path} not found, downloading {name} from {url} ...")
        urllib.request.urlretrieve(url, path)


def canonicalize(smiles: str) -> str | None:
    """One consistent canonicalization algorithm for every source dataset -
    stereochemistry is deliberately ignored (canonical SMILES without stereo
    markers) to keep this first multi-task iteration simpler; revisit if a
    later analysis shows stereo-isomers behaving very differently on any
    task. Strips salts, picks the largest fragment as the parent molecule."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    mol = SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def load_esol() -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(RAW_DIR / "delaney-processed.csv")
    raw["canonical_smiles"] = raw["smiles"].map(canonicalize)
    dropped = int(raw["canonical_smiles"].isna().sum())
    df = raw.dropna(subset=["canonical_smiles"])[["canonical_smiles"]].copy()
    df["solubility"] = raw.loc[df.index, "measured log solubility in mols per litre"]
    return df, dropped


def load_tox21() -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(RAW_DIR / "tox21.csv.gz", compression="gzip")
    raw["canonical_smiles"] = raw["smiles"].map(canonicalize)
    dropped = int(raw["canonical_smiles"].isna().sum())
    df = raw.dropna(subset=["canonical_smiles"])[["canonical_smiles", *TOX21_ASSAYS]].copy()
    return df, dropped


def load_clintox() -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(RAW_DIR / "clintox.csv.gz", compression="gzip")
    raw["canonical_smiles"] = raw["smiles"].map(canonicalize)
    dropped = int(raw["canonical_smiles"].isna().sum())
    df = raw.dropna(subset=["canonical_smiles"])[["canonical_smiles", "FDA_APPROVED", "CT_TOX"]].copy()
    df = df.rename(columns={"FDA_APPROVED": "fda_approved", "CT_TOX": "ct_tox"})
    return df, dropped


def load_bbbp() -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(RAW_DIR / "BBBP.csv")
    raw["canonical_smiles"] = raw["smiles"].map(canonicalize)
    dropped = int(raw["canonical_smiles"].isna().sum())
    df = raw.dropna(subset=["canonical_smiles"])[["canonical_smiles", "p_np"]].copy()
    df = df.rename(columns={"p_np": "bbbp_penetration"})
    return df, dropped


def merge_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge every source on canonical_smiles, averaging regression
    targets and taking the mean (then rounding via nearest-int semantics is
    NOT applied - kept as a float 0/1 average) for binary labels when the
    same molecule appears in a source more than once. Missing task columns
    for a molecule stay NaN - never imputed, per TODO_preprocessing_pipeline.md."""
    merged = None
    for df in sources.values():
        task_cols = [c for c in df.columns if c != "canonical_smiles"]
        agg = df.groupby("canonical_smiles", as_index=False)[task_cols].mean()
        merged = agg if merged is None else merged.merge(agg, on="canonical_smiles", how="outer")
    return merged


def compute_descriptors(smiles_series: pd.Series) -> pd.DataFrame:
    records = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        records.append({name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()})
    return pd.DataFrame(records)


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def scaffold_split(df: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.DataFrame:
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].map(murcko_scaffold)

    rng = np.random.default_rng(seed)
    scaffold_groups = df.groupby("scaffold").indices
    scaffold_keys = list(scaffold_groups.keys())
    rng.shuffle(scaffold_keys)

    n_total = len(df)
    n_train_target = int(ratios[0] * n_total)
    n_val_target = int(ratios[1] * n_total)

    split = np.empty(n_total, dtype=object)
    n_train, n_val = 0, 0
    for scaffold in scaffold_keys:
        idx = scaffold_groups[scaffold]
        if n_train < n_train_target:
            split[idx] = "train"
            n_train += len(idx)
        elif n_val < n_val_target:
            split[idx] = "val"
            n_val += len(idx)
        else:
            split[idx] = "test"

    df["split"] = split
    return df.drop(columns=["scaffold"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ensure_raw_datasets()

    loaders = {"esol": load_esol, "tox21": load_tox21, "clintox": load_clintox, "bbbp": load_bbbp}
    sources, dropped_counts, source_sizes = {}, {}, {}
    for name, loader in loaders.items():
        df, dropped = loader()
        sources[name] = df
        dropped_counts[name] = dropped
        source_sizes[name] = len(df)

    merged = merge_sources(sources)

    descriptors = compute_descriptors(merged["canonical_smiles"])
    full = pd.concat([merged.reset_index(drop=True), descriptors.reset_index(drop=True)], axis=1)
    full = scaffold_split(full, SPLIT_RATIOS, args.seed)

    full.to_csv(PROCESSED_DIR / "admet_processed.csv", index=False)

    # Quality report: per-task label coverage (how many molecules actually
    # have a non-missing label for each task) - expected to be sparse given
    # these 4 datasets barely overlap.
    coverage = {task: int(full[task].notna().sum()) for task in TASK_COLUMNS}
    split_sizes = full["split"].value_counts().to_dict()

    split_smiles = {s: set(g["canonical_smiles"]) for s, g in full.groupby("split")}
    leaks = {}
    splits = list(split_smiles.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_smiles[splits[i]] & split_smiles[splits[j]]
            if overlap:
                leaks[f"{splits[i]}<->{splits[j]}"] = len(overlap)

    report_lines = ["# ADMET multi-task data quality report\n\n"]
    report_lines.append("## Per-source dataset sizes and parsing drops\n\n")
    for name in loaders:
        report_lines.append(f"- {name}: {source_sizes[name]} molecules parsed, {dropped_counts[name]} dropped (unparsable SMILES)\n")
    report_lines.append(f"\n## Merged multi-task table\n\n- Total unique molecules: {len(full)}\n")
    report_lines.append("\n## Per-task label coverage (non-missing labels)\n\n")
    for task in TASK_COLUMNS:
        pct = 100 * coverage[task] / len(full)
        report_lines.append(f"- {task}: {coverage[task]} / {len(full)} ({pct:.1f}%)\n")
    report_lines.append(
        "\nLow coverage per task is expected - these 4 source datasets barely "
        "overlap by molecule, so most rows have a label for only one task. "
        "This is exactly why TODO_preprocessing_pipeline.md calls for masking "
        "(NaN) instead of imputation: a multi-task loss must skip missing "
        "labels per-task per-row, not invent values for them.\n"
    )
    report_lines.append(f"\n## Split sizes\n\n{split_sizes}\n")
    report_lines.append(f"\n## Split leakage (should be empty)\n\n{leaks}\n")

    (PROCESSED_DIR / "report.md").write_text("".join(report_lines), encoding="utf-8")

    print(f"Merged {len(full)} unique molecules from {len(loaders)} sources.")
    print(f"Per-task coverage: {coverage}")
    print(f"Split sizes: {split_sizes}")
    print(f"Split leakage: {leaks}")
    print(f"Wrote: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
