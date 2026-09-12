// Shared constants describing the 16-task ADMET profile shape, matching
// backend-spec/api-contract.md and frontend-spec/components.md /
// data-visualization.md. Centralized here so the radar, cards, and Tox21
// panel all agree on axis order, risk thresholds, and assay labels.

export const EXAMPLE_MOLECULES = [
  { label: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  // Deliberately chosen for its well-known teratogenicity/toxicity profile -
  // a good demo of the toxicity axes actually lighting up (frontend-spec/predict-page.md).
  { label: "Thalidomide", smiles: "C1CC(=O)NC(=O)C1N1C(=O)c2ccccc2C1=O" },
  { label: "Caffeine", smiles: "Cn1cnc2c1c(=O)n(C)c(=O)n2C" },
];

// Human-readable labels for the 12 Tox21 assay codes - see
// frontend-spec/components.md#tox21panelexpander. Full set (only 4 were
// given verbatim in the spec; the rest follow the same style).
export const TOX21_ASSAY_LABELS = {
  "NR-AR": "Androgen receptor activity",
  "NR-AR-LBD": "Androgen receptor (ligand-binding domain) activity",
  "NR-AhR": "Aryl hydrocarbon receptor activity",
  "NR-Aromatase": "Aromatase enzyme inhibition",
  "NR-ER": "Estrogen receptor activity",
  "NR-ER-LBD": "Estrogen receptor (ligand-binding domain) activity",
  "NR-PPAR-gamma": "PPAR-gamma receptor activity",
  "SR-ARE": "Antioxidant response element activation (oxidative stress)",
  "SR-ATAD5": "DNA damage response (genotoxicity)",
  "SR-HSE": "Heat shock response (cellular stress)",
  "SR-MMP": "Mitochondrial membrane potential disruption",
  "SR-p53": "p53 stress response (DNA damage signal)",
};

export const TOX21_ASSAY_ORDER = Object.keys(TOX21_ASSAY_LABELS);

// Risk-color thresholds - frontend-spec/design-system.md. Applied to
// probabilities (0-1); "low" means the low end of that probability, which
// for toxicity-style probabilities means low risk.
export function riskLevel(probability) {
  if (probability >= 0.6) return "high";
  if (probability >= 0.3) return "moderate";
  return "low";
}

export const RISK_COLOR_VAR = {
  low: "var(--risk-low)",
  moderate: "var(--risk-moderate)",
  high: "var(--risk-high)",
};

// Normalization for the 5-axis ADMET radar, see
// frontend-spec/data-visualization.md. Two axes are inverted (1 - value)
// so that "further from center = better/safer" is consistent across all
// 5 axes - otherwise a large polygon area could misleadingly suggest a
// "good" molecule while actually meaning high toxicity.
export function normalizeSolubility(value) {
  return Math.min(1, Math.max(0, (value - -8) / (2 - -8)));
}

export function buildRadarAxes(profile) {
  return [
    {
      key: "solubility",
      label: "Solubility",
      rawValue: profile.solubility.value,
      rawLabel: `${profile.solubility.value.toFixed(2)} ${profile.solubility.unit}`,
      radarValue: normalizeSolubility(profile.solubility.value),
      inverted: false,
    },
    {
      key: "bbb_penetration",
      label: "BBB Penetration",
      rawValue: profile.bbb_penetration.probability,
      rawLabel: `${Math.round(profile.bbb_penetration.probability * 100)}%`,
      radarValue: profile.bbb_penetration.probability,
      inverted: false,
    },
    {
      key: "toxicity_tox21",
      label: "Toxicity (Tox21)",
      rawValue: profile.toxicity_tox21.aggregate_risk,
      rawLabel: `${Math.round(profile.toxicity_tox21.aggregate_risk * 100)}%`,
      radarValue: 1 - profile.toxicity_tox21.aggregate_risk,
      inverted: true,
    },
    {
      key: "clinical_trial_toxicity",
      label: "Clinical Trial Toxicity",
      rawValue: profile.clinical_trial_toxicity.probability,
      rawLabel: `${Math.round(profile.clinical_trial_toxicity.probability * 100)}%`,
      radarValue: 1 - profile.clinical_trial_toxicity.probability,
      inverted: true,
    },
    {
      key: "fda_approval_likelihood",
      label: "FDA Approval Likelihood",
      rawValue: profile.fda_approval_likelihood.probability,
      rawLabel: `${Math.round(profile.fda_approval_likelihood.probability * 100)}%`,
      radarValue: profile.fda_approval_likelihood.probability,
      inverted: false,
    },
  ];
}
