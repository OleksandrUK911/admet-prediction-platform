import numpy as np
import pandas as pd

from ml.preprocess import (
    canonicalize,
    compute_descriptors,
    merge_sources,
    murcko_scaffold,
    scaffold_split,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"


def test_canonicalize_valid_smiles():
    assert canonicalize(ASPIRIN) is not None


def test_canonicalize_invalid_smiles_returns_none():
    assert canonicalize("not-a-smiles!!!") is None


def test_canonicalize_ignores_stereochemistry():
    # Two stereo-variants of the same constitution should canonicalize to
    # the same string, since isomericSmiles=False - a deliberate simplification.
    a = canonicalize("C[C@H](N)C(=O)O")
    b = canonicalize("C[C@@H](N)C(=O)O")
    assert a == b


def test_canonicalize_strips_salts():
    result = canonicalize("CC(=O)[O-].[Na+]")
    assert result is not None
    assert "Na" not in result


def test_canonicalize_equivalent_forms_match():
    a = canonicalize("CCO")
    b = canonicalize("OCC")
    assert a == b


def test_merge_sources_outer_joins_and_masks_missing():
    esol_like = pd.DataFrame({"canonical_smiles": [ETHANOL], "solubility": [-1.0]})
    tox_like = pd.DataFrame({"canonical_smiles": [ASPIRIN], "NR-AR": [1.0]})
    merged = merge_sources({"esol": esol_like, "tox21": tox_like})

    assert len(merged) == 2
    ethanol_row = merged[merged["canonical_smiles"] == ETHANOL].iloc[0]
    aspirin_row = merged[merged["canonical_smiles"] == ASPIRIN].iloc[0]
    assert ethanol_row["solubility"] == -1.0
    assert pd.isna(ethanol_row["NR-AR"])
    assert pd.isna(aspirin_row["solubility"])
    assert aspirin_row["NR-AR"] == 1.0


def test_merge_sources_averages_duplicate_molecule_within_source():
    dup = pd.DataFrame({"canonical_smiles": [ETHANOL, ETHANOL], "solubility": [-1.0, -2.0]})
    merged = merge_sources({"esol": dup})
    assert len(merged) == 1
    assert merged.iloc[0]["solubility"] == -1.5


def test_compute_descriptors_returns_expected_columns():
    descriptors = compute_descriptors(pd.Series([ASPIRIN]))
    assert set(descriptors.columns) == {
        "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount",
    }
    assert descriptors.iloc[0]["RingCount"] == 1


def test_murcko_scaffold_benzene_is_itself():
    assert murcko_scaffold("c1ccccc1") == "c1ccccc1"


def test_scaffold_split_covers_all_rows_no_leakage():
    smiles_list = [ASPIRIN, ETHANOL, "c1ccccc1", "CCC", "CCCCC", "c1ccncc1", "CCN", "CCCl"]
    df = pd.DataFrame({"canonical_smiles": smiles_list, "solubility": [-1.0] * len(smiles_list)})
    result = scaffold_split(df, ratios=(0.5, 0.25, 0.25), seed=42)
    assert set(result["split"].unique()) <= {"train", "val", "test"}
    assert len(result) == len(df)
    assert result["split"].notna().all()


def test_scaffold_split_deterministic_for_same_seed():
    smiles_list = [ASPIRIN, ETHANOL, "c1ccccc1", "CCC", "CCCCC"]
    df = pd.DataFrame({"canonical_smiles": smiles_list, "solubility": [-1.0] * len(smiles_list)})
    a = scaffold_split(df.copy(), ratios=(0.6, 0.2, 0.2), seed=7)
    b = scaffold_split(df.copy(), ratios=(0.6, 0.2, 0.2), seed=7)
    assert a["split"].tolist() == b["split"].tolist()


def test_merge_sources_handles_all_four_task_groups():
    esol_like = pd.DataFrame({"canonical_smiles": [ETHANOL], "solubility": [-1.0]})
    tox_like = pd.DataFrame({"canonical_smiles": [ETHANOL], "NR-AR": [0.0], "SR-p53": [1.0]})
    clintox_like = pd.DataFrame({"canonical_smiles": [ASPIRIN], "fda_approved": [1.0], "ct_tox": [0.0]})
    bbbp_like = pd.DataFrame({"canonical_smiles": [ASPIRIN], "bbbp_penetration": [1.0]})
    merged = merge_sources({
        "esol": esol_like, "tox21": tox_like, "clintox": clintox_like, "bbbp": bbbp_like,
    })
    assert len(merged) == 2
    assert set(merged.columns) >= {
        "canonical_smiles", "solubility", "NR-AR", "SR-p53", "fda_approved", "ct_tox", "bbbp_penetration",
    }
    ethanol_row = merged[merged["canonical_smiles"] == ETHANOL].iloc[0]
    assert np.isclose(ethanol_row["solubility"], -1.0)
    assert pd.isna(ethanol_row["fda_approved"])
