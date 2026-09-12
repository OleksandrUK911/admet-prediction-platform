"""Pydantic request/response schemas, matching backend-spec/api-contract.md
field names exactly."""

from pydantic import BaseModel, Field

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
TOX21_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


class AdmetProfileRequest(BaseModel):
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="A molecule in SMILES notation. Parsed and validated with RDKit; "
        "malformed SMILES are rejected with a 422.",
        examples=[ASPIRIN_SMILES],
    )


class ApplicabilityDomain(BaseModel):
    in_domain: bool = Field(..., description="Whether the molecule falls within the training data's descriptor space.")
    distance_score: float = Field(
        ..., description="Raw nearest-neighbor distance (descriptor-based) to the training set. Not shown directly to users."
    )


class SolubilityProfile(BaseModel):
    value: float = Field(..., description="Predicted log solubility.", examples=[-2.31])
    unit: str = Field("logS", description="Unit of the predicted value.")
    confidence: float | None = Field(
        None, description="Confidence for this prediction. Null: no calibrated uncertainty measure exists for this regression model."
    )


class ProbabilityConfidence(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated classifier probability.")
    confidence: float | None = Field(
        None,
        description="Confidence proxy derived from the calibrated probability's distance from 0.5 "
        "(not a separately validated uncertainty measure).",
    )


class Tox21Profile(BaseModel):
    aggregate_risk: float = Field(
        ..., ge=0.0, le=1.0, description="Max of the 12 Tox21 panel probabilities (worst-case framing)."
    )
    confidence: float | None = Field(None, description="Confidence proxy for the aggregate risk.")
    panel: dict[str, float] = Field(..., description="All 12 Tox21 assay codes mapped to their calibrated probability.")


class AdmetProfile(BaseModel):
    solubility: SolubilityProfile
    bbb_penetration: ProbabilityConfidence
    toxicity_tox21: Tox21Profile
    clinical_trial_toxicity: ProbabilityConfidence
    fda_approval_likelihood: ProbabilityConfidence


class AdmetProfileResponse(BaseModel):
    id: str = Field(..., description="Server-generated UUID identifying this profile.")
    smiles: str = Field(..., description="Canonicalized echo of the input SMILES.", examples=[ASPIRIN_SMILES])
    disclaimer: str = Field(..., description="Fixed research-use disclaimer text.")
    model_version: str = Field(..., description="Version of the production model bundle used.", examples=["0.1.0"])
    applicability_domain: ApplicabilityDomain
    profile: AdmetProfile


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message.", examples=["Could not parse this SMILES string"])


class HistoryItem(BaseModel):
    id: str = Field(..., description="Unique id (UUID4) of this profile.")
    smiles: str = Field(..., description="The canonicalized SMILES that was profiled.", examples=[ASPIRIN_SMILES])
    created_at: str = Field(..., description="UTC timestamp (ISO 8601) when the profile was computed.")


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = Field(..., description="Whether the production model bundle is loaded.")
    db_connected: bool = Field(..., description="Whether the SQLite database is reachable.")
