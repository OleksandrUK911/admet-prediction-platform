import numpy as np
import pytest

from backend.app.inference import MAX_HEAVY_ATOMS, InvalidSmilesError, ModelService
from ml.features import TOX21_ASSAYS

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture(scope="module")
def service():
    return ModelService()


def test_predict_valid_smiles_returns_all_16_tasks(service):
    result = service.predict(ASPIRIN_SMILES)

    assert result["smiles"]  # canonical echo, non-empty
    assert result["disclaimer"]
    assert result["model_version"]
    assert isinstance(result["id"], str) and len(result["id"]) == 36

    profile = result["profile"]
    assert isinstance(profile["solubility"]["value"], float)
    assert profile["solubility"]["unit"] == "logS"

    for key in ("bbb_penetration", "clinical_trial_toxicity", "fda_approval_likelihood"):
        assert 0.0 <= profile[key]["probability"] <= 1.0

    tox21 = profile["toxicity_tox21"]
    assert 0.0 <= tox21["aggregate_risk"] <= 1.0
    assert set(tox21["panel"].keys()) == set(TOX21_ASSAYS)
    assert len(tox21["panel"]) == 12
    for proba in tox21["panel"].values():
        assert 0.0 <= proba <= 1.0
    assert tox21["aggregate_risk"] == max(tox21["panel"].values())


def test_predict_invalid_smiles_raises(service):
    with pytest.raises(InvalidSmilesError):
        service.predict("this is not a smiles string!!!")


def test_predict_is_deterministic(service):
    first = service.predict(ASPIRIN_SMILES)
    second = service.predict(ASPIRIN_SMILES)
    assert first["profile"]["solubility"]["value"] == second["profile"]["solubility"]["value"]
    assert first["profile"]["toxicity_tox21"]["panel"] == second["profile"]["toxicity_tox21"]["panel"]


def test_applicability_domain_present_and_reasonable(service):
    result = service.predict(ASPIRIN_SMILES)
    ad = result["applicability_domain"]
    assert isinstance(ad["in_domain"], bool)
    assert isinstance(ad["distance_score"], float)
    assert ad["distance_score"] >= 0.0


def test_solubility_confidence_is_honestly_null(service):
    """No calibrated per-prediction uncertainty exists for the regression
    model - confidence must be null, not a fabricated number."""
    result = service.predict(ASPIRIN_SMILES)
    assert result["profile"]["solubility"]["confidence"] is None


def test_applicability_domain_in_domain_for_real_molecule(service):
    """A real, ordinary drug-like molecule (aspirin) should fall inside
    the applicability domain - a sanity check for the "normal" case, to
    contrast with the deliberately-out-of-domain case below."""
    result = service.predict(ASPIRIN_SMILES)
    ad = result["applicability_domain"]
    assert ad["in_domain"] is True
    assert ad["distance_score"] < service._ad_threshold


def test_applicability_domain_flags_synthetic_far_outlier(service):
    """Edge case: a descriptor vector far outside anything in the training
    set must be flagged out-of-domain (in_domain=False) - tests the
    flagging logic itself directly (ml/TODO_calibration_uncertainty.md's
    "unit tests for boundary cases"), rather than relying on finding a
    real-world molecule that happens to be an outlier.

    Built by taking the scaler's fitted mean and adding 100 standard
    deviations to every descriptor - guaranteed to be far past the
    OOD_QUANTILE=0.90 threshold distance regardless of dataset specifics."""
    assert service._ad_scaler is not None, "applicability-domain index not built - is data/processed/admet_processed.csv present?"

    n_descriptors = len(service.descriptor_columns)
    far_outlier = service._ad_scaler.mean_ + 100 * np.sqrt(service._ad_scaler.var_)
    ad = service._applicability_domain(far_outlier.reshape(n_descriptors))

    assert ad["in_domain"] is False
    assert ad["distance_score"] > service._ad_threshold


def test_predict_rejects_molecule_over_max_heavy_atoms(service):
    """Defense against adversarial/huge molecules on top of the 300-char
    request-body cap (backend/app/schemas.py) - see
    TODO/backend/TODO_auth_security.md."""
    # A single unbranched carbon chain: cheap to build, trivially exceeds
    # MAX_HEAVY_ATOMS, and still well under the 300-char SMILES length cap.
    long_chain_smiles = "C" * (MAX_HEAVY_ATOMS + 1)
    with pytest.raises(InvalidSmilesError):
        service.predict(long_chain_smiles)


def test_predict_cache_hit_returns_same_profile_but_fresh_id(service):
    """Cache key is (canonical_smiles, model_version) - see
    TODO/backend/TODO_database.md. A repeat call for the same molecule must
    reuse the cached profile/applicability_domain, but never reuse the id -
    each /admet-profile call is its own persisted history row."""
    service.clear_cache()
    first = service.predict(ASPIRIN_SMILES)
    second = service.predict(ASPIRIN_SMILES)

    assert first["id"] != second["id"]
    assert first["profile"] == second["profile"]
    assert first["applicability_domain"] == second["applicability_domain"]


def test_predict_cache_populated_after_first_call(service):
    service.clear_cache()
    result = service.predict(ASPIRIN_SMILES)
    cache_key = (result["smiles"], result["model_version"])
    assert cache_key in service._cache


def test_clear_cache_forces_recomputation(service):
    service.clear_cache()
    service.predict(ASPIRIN_SMILES)
    assert len(service._cache) == 1
    service.clear_cache()
    assert len(service._cache) == 0


def test_applicability_domain_boundary_is_deterministic(service):
    """Same input molecule -> same applicability-domain verdict every time
    (no hidden randomness in the nearest-neighbor lookup)."""
    first = service.predict(ASPIRIN_SMILES)["applicability_domain"]
    second = service.predict(ASPIRIN_SMILES)["applicability_domain"]
    assert first == second
