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

// Comparison-page colors (identity, not risk) - see ../frontend-spec/comparison-page.md
// and ../frontend-spec/design-system.md#порівняння-кольори. Only 3 slots by
// design: the spec caps comparison at 2-3 molecules ("more complicates
// reading overlaid radar charts").
export const COMPARISON_MAX_MOLECULES = 3;
export const COMPARISON_COLORS = ["var(--compare-1)", "var(--compare-2)", "var(--compare-3)"];

// One row per prediction task in the 16-task profile (5 top-level + 12
// Tox21 sub-assays), for the side-by-side comparison table -
// frontend-spec/comparison-page.md + TODO_data_visualization.md ("Side-by-side
// порівняння профілів"). `direction` says which end is "better" for that
// task, so the comparison table can highlight the best/worst molecule per
// row: "higher" for solubility/BBB penetration/FDA approval likelihood
// (matches the non-inverted radar axes), "lower" for every risk
// probability (aggregate Tox21 risk, clinical trial toxicity, and each of
// the 12 individual Tox21 assays) - mirrors the two inverted radar axes.
export const COMPARISON_ROWS = [
  {
    key: "solubility",
    titleKey: "tasks.solubility",
    direction: "higher",
    kind: "regression",
    getValue: (profile) => profile.solubility.value,
    getUnit: (profile) => profile.solubility.unit,
  },
  {
    key: "bbb_penetration",
    titleKey: "tasks.bbbPenetration",
    direction: "higher",
    kind: "probability",
    getValue: (profile) => profile.bbb_penetration.probability,
  },
  {
    key: "toxicity_tox21",
    titleKey: "tasks.toxicityTox21",
    direction: "lower",
    kind: "probability",
    getValue: (profile) => profile.toxicity_tox21.aggregate_risk,
  },
  {
    key: "clinical_trial_toxicity",
    titleKey: "tasks.clinicalTrialToxicity",
    direction: "lower",
    kind: "probability",
    getValue: (profile) => profile.clinical_trial_toxicity.probability,
  },
  {
    key: "fda_approval_likelihood",
    titleKey: "tasks.fdaApprovalLikelihood",
    direction: "higher",
    kind: "probability",
    getValue: (profile) => profile.fda_approval_likelihood.probability,
  },
  ...TOX21_ASSAY_ORDER.map((code) => ({
    key: `tox21_${code}`,
    // No i18n key - same choice as Tox21PanelChart, which shows the
    // plain-English TOX21_ASSAY_LABELS constant untranslated.
    label: `${code} — ${TOX21_ASSAY_LABELS[code]}`,
    direction: "lower",
    kind: "probability",
    getValue: (profile) => profile.toxicity_tox21.panel[code],
  })),
];

// Merges each molecule's 5-axis radar data into one array Recharts can
// plot as multiple overlaid <Radar> series on a single <RadarChart> -
// frontend-spec/comparison-page.md ("Overlaid radar, до 3 профілів").
// `entries` is [{ id, profile }]; the returned rows carry `value_<id>`
// (normalized, for the chart) and `raw_<id>` (human-readable, for the
// tooltip/table fallback) per molecule, keyed by the same axis order as
// buildRadarAxes.
export function buildComparisonRadarData(entries) {
  const perMolecule = entries.map((entry) => buildRadarAxes(entry.profile));
  const axisTemplate = perMolecule[0] || [];
  return axisTemplate.map((axis, axisIndex) => {
    const row = { axisKey: axis.key };
    entries.forEach((entry, i) => {
      row[`value_${entry.id}`] = perMolecule[i][axisIndex].radarValue;
      row[`raw_${entry.id}`] = perMolecule[i][axisIndex].rawLabel;
    });
    return row;
  });
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
