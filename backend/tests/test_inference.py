import pytest

from backend.app.inference import InvalidSmilesError, ModelService
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
