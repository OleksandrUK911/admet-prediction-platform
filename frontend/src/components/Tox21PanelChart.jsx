import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RISK_COLOR_VAR, TOX21_ASSAY_LABELS, TOX21_ASSAY_ORDER, riskLevel } from "../admetConstants";
import { formatPercent } from "../formatters";

function PanelTooltip({ active, payload, language }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", padding: 8, borderRadius: 4 }}>
      <strong>{point.code}</strong> — {point.label}
      <br />
      {formatPercent(point.probability, language)}
    </div>
  );
}

// Tox21 12-assay breakdown, expandable under the main dashboard -
// frontend-spec/components.md#tox21panelexpander and
// frontend-spec/data-visualization.md. Sorted worst-first (highest
// probability at top) and colored by the same risk thresholds as the
// result cards; an accessible "view as table" toggle is the fallback for
// anyone who can't (or would rather not) read the bar chart, mirroring
// project #1's DescriptorBarChart pattern.
export function Tox21PanelChart({ panel, confidence }) {
  const { t, i18n } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const [showTable, setShowTable] = useState(false);

  const data = TOX21_ASSAY_ORDER.map((code) => ({
    code,
    label: TOX21_ASSAY_LABELS[code],
    probability: panel[code],
  })).sort((a, b) => b.probability - a.probability);

  return (
    <div style={{ marginTop: 16, border: "1px solid var(--border)", borderRadius: "var(--radius-card)" }}>
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        aria-controls="tox21-panel-body"
        style={{
          width: "100%",
          textAlign: "left",
          padding: 12,
          background: "var(--surface)",
          border: "none",
          borderRadius: "var(--radius-card)",
          cursor: "pointer",
          color: "var(--text)",
          fontWeight: 600,
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{t("tox21.expanderTitle")}</span>
        <span aria-hidden="true">{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && (
        <div id="tox21-panel-body" style={{ padding: 12 }}>
          {confidence !== null && confidence !== undefined && (
            <p className="text-muted" style={{ fontSize: 12, marginTop: 0 }}>
              {t("tox21.aggregateConfidence", { value: formatPercent(confidence, i18n.language) })}
            </p>
          )}
          <div style={{ width: "100%", height: 320 }}>
            <ResponsiveContainer>
              <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
                <XAxis type="number" domain={[0, 1]} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="code"
                  width={90}
                  tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                />
                <Tooltip content={<PanelTooltip language={i18n.language} />} />
                <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                  {data.map((row) => (
                    <Cell key={row.code} fill={RISK_COLOR_VAR[riskLevel(row.probability)]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <button
            type="button"
            className="icon-button"
            onClick={() => setShowTable((prev) => !prev)}
            aria-expanded={showTable}
            aria-controls="tox21-table"
            style={{ marginTop: 8 }}
          >
            {showTable ? t("chart.hideTable") : t("chart.showTable")}
          </button>
          {showTable && (
            <div className="table-scroll" style={{ marginTop: 8 }}>
              <table id="tox21-table" style={{ width: "100%", borderCollapse: "collapse" }}>
                <caption style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 12, marginBottom: 4 }}>
                  {t("tox21.tableCaption")}
                </caption>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                      {t("tox21.columnAssay")}
                    </th>
                    <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                      {t("tox21.columnLabel")}
                    </th>
                    <th style={{ textAlign: "right", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                      {t("tox21.columnProbability")}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((row) => (
                    <tr key={row.code}>
                      <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>{row.code}</td>
                      <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>{row.label}</td>
                      <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)", textAlign: "right" }}>
                        {formatPercent(row.probability, i18n.language)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
