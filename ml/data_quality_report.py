"""Data quality / validation report for the already-processed ADMET
multi-task table, per TODO/data/TODO_quality_validation.md.

This does NOT re-run preprocessing - it reads data/processed/admet_processed.csv
(written by ml/preprocess.py) and ml/features.py's shared task/descriptor
definitions, and only *analyzes* the result:
  1. Class imbalance for all 15 classification tasks + solubility distribution.
  2. Duplicate-molecule check within each split and across train/val/test -
     the scaffold split in ml/preprocess.py is built to make cross-split
     duplicates impossible, but this verifies that empirically rather than
     assuming it.
  3. Scaffold-level leakage check - no Murcko scaffold should appear in more
     than one split (verified, not assumed) - plus how skewed scaffold
     frequency is (top-N most common scaffolds and how many molecules they
     cover), since real drug-discovery datasets are rarely scaffold-uniform.
  4. Chemical space / scaffold diversity per source dataset (unique
     scaffolds / molecules ratio).
  5. Dataset-wide applicability domain summary: distribution of pairwise
     nearest-neighbor distances in descriptor space within the train split.
     This is a dataset-wide characterization of how "spread out" the
     chemical space is - not a per-task in/out-of-domain check, which
     ml/calibration.py already does for 4 representative tasks.

Usage:
    python ml/data_quality_report.py

Writes ml/results/data_quality_report.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    CLASSIFICATION_TASKS,
    DESCRIPTOR_COLUMNS,
    compute_descriptors,
    task_rows,
)
from preprocess import murcko_scaffold

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "admet_processed.csv"
REPORT_PATH = ROOT / "ml" / "results" / "data_quality_report.md"

TOP_N_SCAFFOLDS = 10

# Proxy mapping from source dataset -> the task column(s) that only that
# source populates (per ml/preprocess.py's loaders). The 4 source datasets
# barely overlap by molecule (see data/processed/report.md's per-task
# coverage), so "has a non-missing label for one of these columns" is a
# reliable stand-in for "came from this source" - the merged table itself
# does not carry a source column (masking is per-task, not per-source).
SOURCE_TASK_COLUMNS = {
    "esol": ["solubility"],
    "tox21": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ],
    "clintox": ["fda_approved", "ct_tox"],
    "bbbp": ["bbbp_penetration"],
}


def class_imbalance_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "## Класовий дисбаланс (15 classification tasks)\n\n",
        (
            "| Task | N labeled | Positives | Negatives | % positive |\n"
            "|---|---|---|---|---|\n"
        ),
    ]
    for task in CLASSIFICATION_TASKS:
        rows = task_rows(df, task)
        y = rows[task].to_numpy()
        n = len(y)
        pos = int(np.sum(y >= 0.5))
        neg = n - pos
        pct = 100 * pos / n if n else float("nan")
        lines.append(f"| {task} | {n} | {pos} | {neg} | {pct:.1f}% |\n")

    sol = df["solubility"].dropna()
    lines.append("\n## Розподіл цільової змінної розчинності (solubility, regression)\n\n")
    lines.append(
        "| N | Min | P25 | Median | P75 | Max | Mean | Std |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines.append(
        f"| {len(sol)} | {sol.min():.2f} | {sol.quantile(0.25):.2f} | "
        f"{sol.median():.2f} | {sol.quantile(0.75):.2f} | {sol.max():.2f} | "
        f"{sol.mean():.2f} | {sol.std():.2f} |\n"
    )
    return lines


def duplicate_check(df: pd.DataFrame) -> tuple[list[str], dict]:
    lines = ["\n## Дублікати молекул усередині та між splits\n\n"]
    findings = {}

    lines.append(
        "### Усередині кожного split\n\n"
        "| Split | N молекул | N унікальних SMILES | N дублікатів |\n|---|---|---|---|\n"
    )
    within_dupes_total = 0
    for split, g in df.groupby("split"):
        n = len(g)
        n_unique = g["canonical_smiles"].nunique()
        n_dupe = n - n_unique
        within_dupes_total += n_dupe
        lines.append(f"| {split} | {n} | {n_unique} | {n_dupe} |\n")
    findings["within_split_duplicates"] = within_dupes_total

    split_smiles = {s: set(g["canonical_smiles"]) for s, g in df.groupby("split")}
    splits = sorted(split_smiles.keys())
    lines.append("\n### Між splits (перетин множин SMILES)\n\n| Split A | Split B | N спільних молекул |\n|---|---|---|\n")
    cross_dupes_total = 0
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_smiles[splits[i]] & split_smiles[splits[j]]
            cross_dupes_total += len(overlap)
            lines.append(f"| {splits[i]} | {splits[j]} | {len(overlap)} |\n")
    findings["cross_split_duplicates"] = cross_dupes_total

    if within_dupes_total == 0 and cross_dupes_total == 0:
        lines.append(
            "\n**Перевірено емпірично: 0 дублікатів усередині кожного split і 0 спільних "
            "молекул між train/val/test.** Це очікувано (кожна унікальна `canonical_smiles` "
            "потрапляє в один рядок після merge за конструкцією `ml/preprocess.py::merge_sources`, "
            "а split призначається на рівні цілого рядка), але перевірено, а не припущено.\n"
        )
    else:
        lines.append(
            f"\n**УВАГА: знайдено {within_dupes_total} дублікатів усередині split(ів) і "
            f"{cross_dupes_total} молекул, що з'являються в кількох splits.** Це суперечить "
            "очікуваній конструкції scaffold split і потребує розслідування в "
            "`ml/preprocess.py`.\n"
        )
    return lines, findings


def scaffold_analysis(df: pd.DataFrame) -> tuple[list[str], dict]:
    lines = ["\n## Scaffold-level leakage та різноманітність\n\n"]
    findings = {}

    scaffolds = df["canonical_smiles"].map(murcko_scaffold)
    df = df.assign(_scaffold=scaffolds)

    scaffold_to_splits = df.groupby("_scaffold")["split"].unique()
    leaking = scaffold_to_splits[scaffold_to_splits.map(len) > 1]
    findings["n_unique_scaffolds"] = int(df["_scaffold"].nunique())
    findings["n_leaking_scaffolds"] = len(leaking)

    lines.append(f"- Загальна кількість молекул: {len(df)}\n")
    lines.append(f"- Унікальних Bemis-Murcko scaffolds: {findings['n_unique_scaffolds']}\n")
    if len(leaking) == 0:
        lines.append(
            "- **Перевірено емпірично: жоден scaffold не зустрічається більш ніж в одному "
            "з train/val/test.** Це очікувано за конструкцією `scaffold_split()` (весь "
            "scaffold-group призначається на один split), але перевірено, а не припущено.\n"
        )
    else:
        lines.append(
            f"- **УВАГА: {len(leaking)} scaffolds зустрічаються в кількох splits** - "
            "це реальний leakage, що суперечить очікуванням scaffold split, потребує "
            "розслідування в `ml/preprocess.py::scaffold_split`.\n"
        )

    counts = df["_scaffold"].value_counts()
    top = counts.head(TOP_N_SCAFFOLDS)
    n_molecules_in_top = int(top.sum())
    pct_in_top = 100 * n_molecules_in_top / len(df)
    findings["top_n_scaffold_molecule_share_pct"] = pct_in_top

    lines.append(
        f"\n### Топ-{TOP_N_SCAFFOLDS} найчастіших scaffolds\n\n"
        f"Реальні датасети для drug discovery часто мають сильно скошений розподіл "
        f"частоти scaffolds (багато аналогів однієї серії сполук) - тому важливо не "
        f"просто порахувати унікальні scaffolds, а й подивитись на розподіл.\n\n"
        "| Rank | Scaffold (SMILES) | N молекул |\n|---|---|---|\n"
    )
    for rank, (scaf, cnt) in enumerate(top.items(), start=1):
        display = scaf if scaf else "(порожній - ациклічна молекула)"
        if len(display) > 60:
            display = display[:57] + "..."
        lines.append(f"| {rank} | `{display}` | {cnt} |\n")
    lines.append(
        f"\nНа топ-{TOP_N_SCAFFOLDS} scaffolds припадає {n_molecules_in_top} молекул "
        f"({pct_in_top:.1f}% усього датасету) - "
        + (
            "помітно скошений розподіл, типовий для ADMET/drug-discovery даних.\n"
            if pct_in_top > 5
            else "розподіл відносно рівномірний, скошеність невисока для цього датасету.\n"
        )
    )

    # Diversity per source (proxy via non-missing task columns, since the
    # merged table has no explicit source column - see SOURCE_TASK_COLUMNS).
    lines.append(
        "\n### Хімічна різноманітність за джерелом датасету (unique scaffolds / molecules)\n\n"
        "Приналежність до джерела визначається за тим, які task-колонки заповнені "
        "(джерела майже не перетинаються за молекулами - див. `data/processed/report.md`), "
        "оскільки об'єднана таблиця не зберігає окрему колонку джерела.\n\n"
        "| Джерело | N молекул | N унікальних scaffolds | Diversity ratio |\n|---|---|---|---|\n"
    )
    diversity = {}
    for source, cols in SOURCE_TASK_COLUMNS.items():
        mask = df[cols].notna().any(axis=1)
        sub = df[mask]
        n = len(sub)
        n_scaf = sub["_scaffold"].nunique()
        ratio = n_scaf / n if n else float("nan")
        diversity[source] = {"n_molecules": n, "n_scaffolds": n_scaf, "ratio": ratio}
        lines.append(f"| {source} | {n} | {n_scaf} | {ratio:.3f} |\n")
    findings["diversity_by_source"] = diversity
    lines.append(
        "\nDiversity ratio ближче до 1.0 означає, що майже кожна молекула має унікальний "
        "scaffold (висока структурна різноманітність); значення набагато нижче 1.0 означає "
        "багато молекул на один scaffold (сполуки-аналоги, congeneric series).\n"
    )

    return lines, findings


def applicability_domain_summary(df: pd.DataFrame) -> list[str]:
    lines = [
        (
            "\n## Applicability domain: загальна характеристика хімічного простору train\n\n"
            "Це доповнює per-task аналіз у `ml/calibration.py` (nearest-neighbor distance "
            "test-vs-train для 4 задач) - тут натомість характеризується сам train split "
            "загалом: наскільки він \"розсіяний\" чи \"кластеризований\" у просторі 7 спільних "
            "дескрипторів (`ml/features.py::DESCRIPTOR_COLUMNS`), незалежно від жодної задачі.\n\n"
        )
    ]

    train = df[df["split"] == "train"]
    descriptors = compute_descriptors(train["canonical_smiles"])
    X = descriptors[DESCRIPTOR_COLUMNS].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    # k=2: nearest neighbor excluding the point itself (index 0 is self).
    nn = NearestNeighbors(n_neighbors=2).fit(X_scaled)
    dist, _ = nn.kneighbors(X_scaled)
    nn_dist = dist[:, 1]

    lines.append(
        f"Попарна відстань (Euclidean, у масштабованому 7-вимірному дескрипторному "
        f"просторі) до найближчого сусіда всередині train split (N={len(train)}):\n\n"
        "| Min | P25 | Median | P75 | P90 | Max | Mean | Std |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    q = np.quantile(nn_dist, [0.25, 0.5, 0.75, 0.9])
    lines.append(
        f"| {nn_dist.min():.3f} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | {q[3]:.3f} | "
        f"{nn_dist.max():.3f} | {nn_dist.mean():.3f} | {nn_dist.std():.3f} |\n"
    )
    skew_ratio = q[3] / q[1] if q[1] > 0 else float("inf")
    lines.append(
        f"\nСпіввідношення P90/median найближчої відстані ({skew_ratio:.2f}) - чим більше "
        "воно за 1, тим сильніше train розділений на щільні кластери (типові молекули) і "
        "розріджені \"хвости\" (нетипові/малочисельні структурні класи), для яких надійність "
        "прогнозу нижча. Це узгоджується зі скошеним розподілом scaffold-частоти вище: "
        "щільні скупчення в дескрипторному просторі відповідають великим congeneric series.\n"
    )
    return lines


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)

    lines = ["# ADMET data quality & validation report\n\n"]
    lines.append(f"Джерело даних: `{PROCESSED_CSV.relative_to(ROOT)}` ({len(df)} молекул).\n\n")

    lines.extend(class_imbalance_table(df))
    dup_lines, dup_findings = duplicate_check(df)
    lines.extend(dup_lines)
    scaf_lines, scaf_findings = scaffold_analysis(df)
    lines.extend(scaf_lines)
    lines.extend(applicability_domain_summary(df))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print(f"Within-split duplicates: {dup_findings['within_split_duplicates']}")
    print(f"Cross-split duplicates: {dup_findings['cross_split_duplicates']}")
    print(f"Unique scaffolds: {scaf_findings['n_unique_scaffolds']}")
    print(f"Leaking scaffolds (should be 0): {scaf_findings['n_leaking_scaffolds']}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
