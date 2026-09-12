import { useTranslation } from "react-i18next";
import { formatPercent } from "../formatters";

// Inline confidence visualization for AdmetResultCard - a shaded band
// around the point value rather than a bare percentage, per
// frontend-spec/data-visualization.md ("confidence-бар (inline)") and the
// project-wide principle in TODO_ui_components.md: never show a bare
// percentage without conveying uncertainty.
//
// The API gives a single calibration-derived `confidence` in [0,1], not a
// formal statistical confidence interval - so the shaded band's width is a
// deliberately-labeled VISUAL PROXY (half-width = (1 - confidence) / 2 of
// the 0-100% scale), not a true confidence interval. This is called out in
// the tooltip and in the About page's "How confidence works" section.
export function ConfidenceBar({ confidence }) {
  const { t, i18n } = useTranslation();

  if (confidence === null || confidence === undefined) {
    return <p className="text-muted" style={{ fontSize: 12, margin: "4px 0 0" }}>{t("card.noConfidence")}</p>;
  }

  const halfWidthPct = ((1 - confidence) / 2) * 100;
  const centerPct = confidence * 100;
  const bandStart = Math.max(0, centerPct - halfWidthPct);
  const bandWidth = Math.min(100, centerPct + halfWidthPct) - bandStart;

  return (
    <div style={{ marginTop: 8 }} title={t("card.confidenceTooltip")}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: "var(--text-muted)",
          marginBottom: 2,
        }}
      >
        <span>{t("card.confidenceLabel")}</span>
        <span>{formatPercent(confidence, i18n.language)}</span>
      </div>
      <div
        role="img"
        aria-label={t("card.confidenceAriaLabel", { value: formatPercent(confidence, i18n.language) })}
        style={{
          position: "relative",
          height: 8,
          borderRadius: 4,
          background: "var(--border)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `${bandStart}%`,
            width: `${bandWidth}%`,
            top: 0,
            bottom: 0,
            background: "var(--accent)",
            opacity: 0.35,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `${centerPct}%`,
            width: 2,
            top: -1,
            bottom: -1,
            background: "var(--accent)",
            transform: "translateX(-1px)",
          }}
        />
      </div>
    </div>
  );
}
