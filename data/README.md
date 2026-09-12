# Datasets — ADMET Prediction

Four MoleculeNet datasets, one per ADMET endpoint group, per
`TODO/data/TODO_sources_licensing.md`'s plan. Same download approach as
project #1 (self-downloading via a script, not a manual step — see
`ml/preprocess.py`).

## 1. Solubility — ESOL (Delaney)
- Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv`
- 1128 compounds. Target: `measured log solubility in mols per litre`.
- Same dataset used in project #1 — reused here rather than switching to
  AqSolDB, for consistency and because it's already validated/well-understood.
- Citation: Delaney, J. S. "ESOL: Estimating Aqueous Solubility Directly from
  Molecular Structure." J. Chem. Inf. Comput. Sci. 44.3 (2004).

## 2. Toxicity panel — Tox21
- Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`
- 7831 compounds, 12 binary assay columns (nuclear receptor / stress response):
  `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`,
  `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`.
  Many missing labels per-compound (not every compound tested against every
  assay) — expected, handled via masking, not imputation, per
  `TODO/data/TODO_preprocessing_pipeline.md`.
- These 12 codes exactly match what's already documented in
  `backend-spec/api-contract.md`'s `toxicity_tox21.panel` field — confirmed
  consistent, no contract update needed.
- Source: Tox21 Data Challenge (NIH/EPA/NCATS), public domain (US government
  work).

## 3. Clinical trial toxicity / FDA approval — ClinTox
- Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz`
- 1484 compounds, 2 binary columns: `FDA_APPROVED`, `CT_TOX`.
- Matches `backend-spec/api-contract.md`'s `clinical_trial_toxicity` /
  `fda_approval_likelihood` fields exactly.
- Source: derived from ClinicalTrials.gov + FDA data via MoleculeNet,
  permissive for research/educational reuse.

## 4. Blood-brain barrier penetration — BBBP
- Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv`
  (note: capitalized filename — lowercase `bbbp.csv`/`.csv.gz` return 403 on
  this mirror)
- 2050 compounds, target column `p_np` (1 = penetrates, 0 = does not).
- Matches `backend-spec/api-contract.md`'s `bbb_penetration` field.

## Licensing

All four are MoleculeNet-distributed datasets under permissive terms for
research/educational reuse, consistent with project #1's ESOL license
finding. Tox21 specifically is US government public-domain data. Cite the
original sources above when publishing results derived from this data.
Non-commercial, research/educational use only — matches this project's
"research/educational" disclaimer in `README.md`.

## Known complexity ahead (Sprint 1)

Merging four datasets of very different sizes (1128–7831 compounds) into
one multi-task table by canonical SMILES will produce a LOT of missing
labels per task (e.g. a compound in ESOL almost certainly wasn't tested in
Tox21) — this is expected and exactly why `TODO/data/TODO_preprocessing_pipeline.md`
calls for masking (NaN) rather than imputing missing labels for the
classification tasks.
