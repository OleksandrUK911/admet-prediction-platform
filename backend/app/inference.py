"""Inference logic, isolated from FastAPI so it can be unit-tested and
reused without spinning up the web layer (same convention as project #1's
backend/app/inference.py).

Field names in the returned dict match backend-spec/api-contract.md's
POST /admet-profile response exactly.

Reuses ml/features.py's compute_descriptors / compute_fingerprints EXACTLY
as training did, so inference-time features match training-time features.
"""

import json
import uuid
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.features import TOX21_ASSAYS, compute_descriptors, compute_fingerprints

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_DIR = REPO_ROOT / "models" / "production"
PROCESSED_CSV = REPO_ROOT / "data" / "processed" / "admet_processed.csv"

DISCLAIMER = "Research/educational use only. Not for clinical or regulatory decision-making."

# Top 10% nearest-neighbor distance (in scaled 7-descriptor space) to the
# training set = "out of domain", matching the threshold convention already
# used in ml/calibration.py's applicability-domain check.
OOD_QUANTILE = 0.90


class InvalidSmilesError(ValueError):
    """Raised when RDKit cannot parse the input SMILES string."""


class TaskProbability(TypedDict):
    probability: float
    confidence: float | None


class SolubilityResult(TypedDict):
    value: float
    unit: str
    confidence: float | None


class Tox21Result(TypedDict):
    aggregate_risk: float
    confidence: float | None
    panel: dict[str, float]


class ApplicabilityDomain(TypedDict):
    in_domain: bool
    distance_score: float


class PredictionResult(TypedDict):
    id: str
    smiles: str
    disclaimer: str
    model_version: str
    applicability_domain: ApplicabilityDomain
    profile: dict


class ModelService:
    """Loads all 16 per-task production models once and serves predictions.
    Instantiate a single instance at application startup - do not reload
    per request (see backend/app/main.py's lifespan)."""

    def __init__(self, production_dir: Path = PRODUCTION_DIR, processed_csv: Path = PROCESSED_CSV):
        models_path = production_dir / "models.joblib"
        metadata_path = production_dir / "metadata.json"
        if not models_path.exists():
            raise FileNotFoundError(
                f"No production models at {models_path}. Run ml/preprocess.py, ml/baseline.py, "
                "ml/experiments_per_task_models.py, and ml/model_registry.py first."
            )
        bundle = joblib.load(models_path)
        self.tasks: dict = bundle["tasks"]
        self.descriptor_columns: list[str] = bundle["descriptor_columns"]
        self.metadata: dict = json.loads(metadata_path.read_text(encoding="utf-8"))

        # Applicability domain: nearest-neighbor distance (in scaled
        # descriptor space) to the training set, same technique as
        # ml/calibration.py. Fit the scaler/NN index on ALL rows of the
        # processed dataset (not just one task's train split) - simpler
        # than picking one task's split, and a reasonable proxy for "have
        # we seen chemistry like this at all" across every task, since the
        # descriptor space (not task labels) is what applicability domain
        # measures here. Documented simplification, see backend-spec.
        self._ad_scaler: StandardScaler | None = None
        self._ad_nn: NearestNeighbors | None = None
        self._ad_threshold: float | None = None
        if processed_csv.exists():
            df = pd.read_csv(processed_csv)
            train_descriptors = compute_descriptors(df["canonical_smiles"])[self.descriptor_columns].to_numpy()
            self._ad_scaler = StandardScaler().fit(train_descriptors)
            scaled = self._ad_scaler.transform(train_descriptors)
            # The index used to serve real queries only needs k=1 (nearest
            # training neighbor to a NEW molecule).
            self._ad_nn = NearestNeighbors(n_neighbors=1).fit(scaled)
            # To calibrate the threshold we instead measure each training
            # point's distance to its nearest OTHER training point (k=2,
            # keep column 1) - querying the same set with k=1 would just
            # return each point matching itself at distance 0 and the
            # threshold would always be 0.
            self_nn = NearestNeighbors(n_neighbors=2).fit(scaled)
            self_dist, _ = self_nn.kneighbors(scaled)
            self._ad_threshold = float(np.quantile(self_dist[:, 1], OOD_QUANTILE))

    def _applicability_domain(self, descriptor_row: np.ndarray) -> ApplicabilityDomain:
        if self._ad_scaler is None or self._ad_nn is None or self._ad_threshold is None:
            # No processed dataset available (e.g. minimal test fixture) -
            # can't compute a real distance, so be honest rather than fake it.
            return {"in_domain": True, "distance_score": 0.0}
        scaled = self._ad_scaler.transform(descriptor_row.reshape(1, -1))
        dist, _ = self._ad_nn.kneighbors(scaled)
        distance_score = float(dist[0, 0])
        return {"in_domain": distance_score < self._ad_threshold, "distance_score": distance_score}

    def _predict_task(self, task: str, descriptor_row: np.ndarray, fingerprint_row: np.ndarray) -> tuple[float, str]:
        """Returns (value, task_type) for one task - probability for
        classification, raw predicted value for regression."""
        bundle = self.tasks[task]
        model = bundle["model"]
        feature_type = bundle["feature_type"]
        X = fingerprint_row.reshape(1, -1) if feature_type == "fingerprints" else descriptor_row.reshape(1, -1)

        if bundle["task_type"] == "classification":
            proba = float(model.predict_proba(X)[0, 1])
            return proba, "classification"

        scaler = bundle.get("scaler")
        X_in = scaler.transform(X) if scaler is not None else X
        value = float(model.predict(X_in)[0])
        return value, "regression"

    def predict(self, smiles: str) -> PredictionResult:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise InvalidSmilesError("Could not parse this SMILES string")

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        smiles_series = pd.Series([canonical_smiles])
        descriptor_row = compute_descriptors(smiles_series)[self.descriptor_columns].to_numpy()[0]
        fingerprint_row = compute_fingerprints(smiles_series)[0]

        # Confidence: for classification tasks we surface the calibrated
        # classifier's own probability-of-the-predicted-class as a rough
        # confidence proxy (a probability near 0 or 1 = the model is more
        # sure; near 0.5 = a coin flip) - this is NOT a separate, validated
        # uncertainty estimate (e.g. a prediction interval or ensemble
        # variance), just a repurposing of the Platt-calibrated probability
        # itself. For the regression task (solubility) there is no
        # equivalent single-model uncertainty measure available (no
        # ensemble spread, no per-prediction interval was fit) - honest
        # choice is `None` rather than a fabricated number, per the task
        # brief and matching project #1's precedent for un-calibrated
        # point predictions.
        def classification_confidence(proba: float) -> float:
            return round(max(proba, 1 - proba), 4)

        solubility_value, _ = self._predict_task("solubility", descriptor_row, fingerprint_row)
        solubility: SolubilityResult = {"value": round(solubility_value, 4), "unit": "logS", "confidence": None}

        bbbp_proba, _ = self._predict_task("bbbp_penetration", descriptor_row, fingerprint_row)
        bbb_penetration: TaskProbability = {
            "probability": round(bbbp_proba, 4),
            "confidence": classification_confidence(bbbp_proba),
        }

        panel: dict[str, float] = {}
        for assay in TOX21_ASSAYS:
            proba, _ = self._predict_task(assay, descriptor_row, fingerprint_row)
            panel[assay] = round(proba, 4)
        aggregate_risk = max(panel.values())
        toxicity_tox21: Tox21Result = {
            "aggregate_risk": round(aggregate_risk, 4),
            "confidence": classification_confidence(aggregate_risk),
            "panel": panel,
        }

        ct_tox_proba, _ = self._predict_task("ct_tox", descriptor_row, fingerprint_row)
        clinical_trial_toxicity: TaskProbability = {
            "probability": round(ct_tox_proba, 4),
            "confidence": classification_confidence(ct_tox_proba),
        }

        fda_proba, _ = self._predict_task("fda_approved", descriptor_row, fingerprint_row)
        fda_approval_likelihood: TaskProbability = {
            "probability": round(fda_proba, 4),
            "confidence": classification_confidence(fda_proba),
        }

        return {
            "id": str(uuid.uuid4()),
            "smiles": canonical_smiles,
            "disclaimer": DISCLAIMER,
            "model_version": self.metadata.get("model_version", "unknown"),
            "applicability_domain": self._applicability_domain(descriptor_row),
            "profile": {
                "solubility": solubility,
                "bbb_penetration": bbb_penetration,
                "toxicity_tox21": toxicity_tox21,
                "clinical_trial_toxicity": clinical_trial_toxicity,
                "fda_approval_likelihood": fda_approval_likelihood,
            },
        }
