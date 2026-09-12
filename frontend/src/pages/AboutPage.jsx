import { useTranslation } from "react-i18next";

// Static content page - frontend-spec/about-page.md. No loading/error
// states by design (nothing is fetched). Wording for datasets/limitations
// is sourced from models/production/metadata.json's known_limitations and
// ml/results/full_evaluation_report.md's overfitting-gap check, not
// invented - see locales/en.json for the exact strings.
export function AboutPage() {
  const { t } = useTranslation();

  const datasetRows = [
    { key: "solubility", nameKey: "about.datasets.solubility" },
    { key: "bbb", nameKey: "about.datasets.bbb" },
    { key: "tox21", nameKey: "about.datasets.tox21" },
    { key: "clintox", nameKey: "about.datasets.clintox" },
  ];

  return (
    <div>
      <h1>{t("about.title")}</h1>

      <h2>{t("about.whatThisIsTitle")}</h2>
      <p>{t("about.whatThisIsText")}</p>

      <h2>{t("about.datasetsTitle")}</h2>
      <div className="table-scroll">
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 480 }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                {t("about.columnTask")}
              </th>
              <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                {t("about.columnDataset")}
              </th>
            </tr>
          </thead>
          <tbody>
            {datasetRows.map((row) => (
              <tr key={row.key}>
                <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>
                  {t(`about.taskNames.${row.key}`)}
                </td>
                <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>{t(row.nameKey)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-muted" style={{ fontSize: 12 }}>
        {t("about.datasetsSourceNote")}
      </p>

      <h2>{t("about.confidenceTitle")}</h2>
      <p>{t("about.confidenceText")}</p>
      <p>{t("about.applicabilityDomainText")}</p>

      <h2>{t("about.limitationsTitle")}</h2>
      <ul>
        <li>{t("about.limitations.overfitting")}</li>
        <li>{t("about.limitations.perTaskModels")}</li>
        <li>{t("about.limitations.calibration")}</li>
        <li>{t("about.limitations.datasetOverlap")}</li>
        <li>{t("about.limitations.tox21Coverage")}</li>
        <li>{t("about.limitations.applicabilityDomainHeuristic")}</li>
      </ul>

      <h2>{t("about.disclaimerTitle")}</h2>
      <p className="text-muted">{t("about.disclaimerText")}</p>
    </div>
  );
}
