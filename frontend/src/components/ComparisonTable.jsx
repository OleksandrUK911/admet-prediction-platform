import { useTranslation } from "react-i18next";
import { COMPARISON_ROWS } from "../admetConstants";
import { formatNumber, formatPercent } from "../formatters";

function formatCellValue(row, value, unit, language) {
  if (row.kind === "probability") return formatPercent(value, language);
  return `${formatNumber(value, language)}${unit ? ` ${unit}` : ""}`;
}

// Side-by-side diff table for all 16 profile tasks (5 top-level + 12
// Tox21 assays) - frontend-spec/comparison-page.md ("Таблиця відмінностей")
// + TODO_data_visualization.md ("Візуальне виділення відмінностей між
// молекулами при порівнянні"). The best/worst cell per row is marked with
// an icon AND a visually-hidden text label, not color alone - same
// accessibility pattern as AdmetResultCard's risk badges
// (TODO_accessibility_responsive.md). A molecule that failed to resolve
// (invalid SMILES / server error) shows "—" in its column instead of
// blocking the whole comparison - frontend-spec/comparison-page.md's
// "Стани" table ("часткова відмова, не блокує все порівняння").
export function ComparisonTable({ molecules }) {
  const { t, i18n } = useTranslation();

  return (
    <div className="table-scroll">
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 520 }}>
        <caption style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 12, marginBottom: 4 }}>
          {t("compare.tableCaption")}
        </caption>
        <thead>
          <tr>
            <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "6px 8px" }}>
              {t("compare.columnEndpoint")}
            </th>
            {molecules.map((m) => (
              <th
                key={m.index}
                style={{
                  textAlign: "left",
                  borderBottom: "1px solid var(--border)",
                  padding: "6px 8px",
                  color: m.color,
                }}
              >
                <span aria-hidden="true">{"●"} </span>
                {m.label}
              </th>
            ))}
            <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "6px 8px" }}>
              {t("compare.columnLargestDiff")}
            </th>
          </tr>
        </thead>
        <tbody>
          {COMPARISON_ROWS.map((row) => {
            const cellValues = molecules.map((m) => (m.status === "success" ? row.getValue(m.profile) : null));
            const validValues = cellValues.filter((v) => v !== null && v !== undefined);
            const hasSpread = validValues.length >= 2 && new Set(validValues).size > 1;
            const best = hasSpread ? (row.direction === "higher" ? Math.max(...validValues) : Math.min(...validValues)) : null;
            const worst = hasSpread ? (row.direction === "higher" ? Math.min(...validValues) : Math.max(...validValues)) : null;
            const successfulProfile = molecules.find((m) => m.status === "success")?.profile;
            const unit = row.kind === "regression" && successfulProfile ? row.getUnit(successfulProfile) : null;
            const spread = validValues.length >= 2 ? Math.max(...validValues) - Math.min(...validValues) : null;

            return (
              <tr key={row.key}>
                <th scope="row" style={{ textAlign: "left", fontWeight: 400, padding: "6px 8px", borderBottom: "1px solid var(--border)" }}>
                  {row.titleKey ? t(row.titleKey) : row.label}
                </th>
                {molecules.map((m, i) => {
                  const raw = cellValues[i];
                  if (m.status === "error") {
                    return (
                      <td key={m.index} style={{ padding: "6px 8px", borderBottom: "1px solid var(--border)" }}>
                        <span aria-hidden="true">{"⚠"} </span>
                        {"—"}
                        <span className="visually-hidden">{t("compare.errorCell")}</span>
                      </td>
                    );
                  }
                  if (m.status !== "success") {
                    return (
                      <td key={m.index} style={{ padding: "6px 8px", borderBottom: "1px solid var(--border)" }} className="text-muted">
                        {t("compare.pendingCell")}
                      </td>
                    );
                  }
                  const isBest = best !== null && raw === best;
                  const isWorst = !isBest && worst !== null && raw === worst;
                  return (
                    <td
                      key={m.index}
                      style={{
                        padding: "6px 8px",
                        borderBottom: "1px solid var(--border)",
                        fontWeight: isBest || isWorst ? 700 : 400,
                        background: isBest
                          ? "color-mix(in srgb, var(--risk-low) 18%, transparent)"
                          : isWorst
                            ? "color-mix(in srgb, var(--risk-high) 14%, transparent)"
                            : undefined,
                      }}
                    >
                      {isBest && <span aria-hidden="true">{"✓"} </span>}
                      {isWorst && <span aria-hidden="true">{"⚠"} </span>}
                      {formatCellValue(row, raw, unit, i18n.language)}
                      {isBest && <span className="visually-hidden"> {t("compare.bestCell")}</span>}
                      {isWorst && <span className="visually-hidden"> {t("compare.worstCell")}</span>}
                    </td>
                  );
                })}
                <td style={{ padding: "6px 8px", borderBottom: "1px solid var(--border)" }} className="text-muted">
                  {spread !== null ? formatCellValue(row, spread, unit, i18n.language) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
