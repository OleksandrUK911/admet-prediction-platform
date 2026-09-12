import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { buildRadarAxes } from "../admetConstants";

function RadarTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", padding: 8, borderRadius: 4 }}>
      <strong>{point.label}</strong>: {point.rawLabel}
    </div>
  );
}

// 5-axis ADMET radar - frontend-spec/data-visualization.md. Two axes
// (toxicity, clinical trial toxicity) are pre-inverted in buildRadarAxes so
// "further from center" consistently means "better/safer" on every axis;
// the tooltip and the "view as table" fallback both show the original,
// non-inverted value so nothing is lost.
export function AdmetRadarChart({ profile }) {
  const { t } = useTranslation();
  const [showTable, setShowTable] = useState(false);
  const axes = buildRadarAxes(profile);
  const data = axes.map((axis) => ({ ...axis, label: t(`radar.${axis.key}`) }));

  return (
    <div>
      <div style={{ width: "100%", height: 280 }}>
        <ResponsiveContainer>
          <RadarChart data={data} outerRadius="75%">
            <PolarGrid stroke="var(--border)" />
            <PolarAngleAxis dataKey="label" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
            <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
            <Radar dataKey="radarValue" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.35} />
            <Tooltip content={<RadarTooltip />} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <button
        type="button"
        className="icon-button"
        onClick={() => setShowTable((prev) => !prev)}
        aria-expanded={showTable}
        aria-controls="radar-table"
        style={{ marginTop: 8 }}
      >
        {showTable ? t("chart.hideTable") : t("chart.showTable")}
      </button>
      {showTable && (
        <div className="table-scroll" style={{ marginTop: 8 }}>
          <table id="radar-table" style={{ width: "100%", borderCollapse: "collapse" }}>
            <caption style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 12, marginBottom: 4 }}>
              {t("radar.tableCaption")}
            </caption>
            <thead>
              <tr>
                <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                  {t("radar.columnAxis")}
                </th>
                <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                  {t("radar.columnValue")}
                </th>
              </tr>
            </thead>
            <tbody>
              {data.map((row) => (
                <tr key={row.key}>
                  <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>{row.label}</td>
                  <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>{row.rawLabel}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
