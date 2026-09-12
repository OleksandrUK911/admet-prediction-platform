import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { buildComparisonRadarData } from "../admetConstants";

function ComparisonRadarTooltip({ active, payload, molecules, label }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", padding: 8, borderRadius: 4 }}>
      <strong>{label}</strong>
      <ul style={{ margin: "4px 0 0", padding: 0, listStyle: "none" }}>
        {molecules.map((m) => (
          <li key={m.index} style={{ color: m.color }}>
            {m.label}: {point[`raw_${m.index}`]}
          </li>
        ))}
      </ul>
    </div>
  );
}

// Overlaid multi-molecule radar - frontend-spec/comparison-page.md
// ("Overlaid radar (до 3 профілів, різні кольори)") + TODO_data_visualization.md
// ("Side-by-side порівняння профілів кількох молекул"). Extends the same
// 5-axis normalization as the single-molecule AdmetRadarChart
// (buildComparisonRadarData reuses buildRadarAxes per molecule) so the two
// charts stay visually consistent; deliberately no confidence band on the
// overlay (would be unreadable with up to 3 semi-transparent polygons) -
// per-molecule confidence stays in that molecule's own AdmetResultCard.
export function AdmetComparisonRadarChart({ molecules }) {
  const { t } = useTranslation();
  const [showTable, setShowTable] = useState(false);

  const rows = buildComparisonRadarData(molecules.map((m) => ({ id: m.index, profile: m.profile })));
  const data = rows.map((row) => ({ ...row, label: t(`radar.${row.axisKey}`) }));

  return (
    <div>
      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <RadarChart data={data} outerRadius="70%">
            <PolarGrid stroke="var(--border)" />
            <PolarAngleAxis dataKey="label" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
            <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
            {molecules.map((m) => (
              <Radar
                key={m.index}
                name={m.label}
                dataKey={`value_${m.index}`}
                stroke={m.color}
                fill={m.color}
                fillOpacity={0.25}
              />
            ))}
            <Legend />
            <Tooltip content={<ComparisonRadarTooltip molecules={molecules} />} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <button
        type="button"
        className="icon-button"
        onClick={() => setShowTable((prev) => !prev)}
        aria-expanded={showTable}
        aria-controls="comparison-radar-table"
        style={{ marginTop: 8 }}
      >
        {showTable ? t("chart.hideTable") : t("chart.showTable")}
      </button>
      {showTable && (
        <div className="table-scroll" style={{ marginTop: 8 }}>
          <table id="comparison-radar-table" style={{ width: "100%", borderCollapse: "collapse" }}>
            <caption style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 12, marginBottom: 4 }}>
              {t("radar.tableCaption")}
            </caption>
            <thead>
              <tr>
                <th style={{ textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 8px" }}>
                  {t("radar.columnAxis")}
                </th>
                {molecules.map((m) => (
                  <th
                    key={m.index}
                    style={{
                      textAlign: "left",
                      borderBottom: "1px solid var(--border)",
                      padding: "4px 8px",
                      color: m.color,
                    }}
                  >
                    {m.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row) => (
                <tr key={row.axisKey}>
                  <td style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>{row.label}</td>
                  {molecules.map((m) => (
                    <td key={m.index} style={{ padding: "4px 8px", borderBottom: "1px solid var(--border)" }}>
                      {row[`raw_${m.index}`]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
