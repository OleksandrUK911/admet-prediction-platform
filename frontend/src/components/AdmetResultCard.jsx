import { useTranslation } from "react-i18next";
import { RISK_COLOR_VAR, riskLevel } from "../admetConstants";
import { ConfidenceBar } from "./ConfidenceBar";

// One card per profile endpoint - frontend-spec/components.md#admetresultcard.
// `probability` cards get a risk badge (colored per design-system.md
// thresholds, always paired with a text label so risk is never color-only -
// TODO_accessibility_responsive.md); the regression `solubility` card shows
// its value+unit with no risk badge (it isn't a probability).
export function AdmetResultCard({ titleKey, valueLabel, probability, confidence, outOfDomain }) {
  const { t } = useTranslation();
  const level = probability === null || probability === undefined ? null : riskLevel(probability);

  return (
    <div
      style={{
        position: "relative",
        padding: 16,
        borderRadius: "var(--radius-card)",
        background: "var(--surface)",
        border: "1px solid var(--border)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <h3 style={{ margin: 0 }}>{t(titleKey)}</h3>
        {level && (
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: "white",
              background: RISK_COLOR_VAR[level],
              padding: "2px 8px",
              borderRadius: "var(--radius-chip)",
              whiteSpace: "nowrap",
            }}
          >
            {t(`riskLevel.${level}`)}
          </span>
        )}
      </div>
      <p style={{ fontSize: 22, fontWeight: 600, margin: "8px 0 0" }}>{valueLabel}</p>
      <ConfidenceBar confidence={confidence} />
      {outOfDomain && (
        <div
          role="note"
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            textAlign: "center",
            padding: 12,
            background: "color-mix(in srgb, var(--surface) 55%, transparent)",
            backdropFilter: "blur(1px)",
            borderRadius: "var(--radius-card)",
            fontSize: 12,
            fontWeight: 600,
            color: "var(--risk-moderate)",
          }}
        >
          <span aria-hidden="true">{"⚠️"} </span>
          {t("applicabilityDomain.cardOverlay")}
        </div>
      )}
    </div>
  );
}
