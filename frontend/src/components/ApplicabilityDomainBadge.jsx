import { useTranslation } from "react-i18next";

// frontend-spec/components.md#applicabilitydomainbadge. Never relies on
// color alone (TODO_accessibility_responsive.md) - icon + text always
// carry the meaning too.
export function ApplicabilityDomainBadge({ inDomain }) {
  const { t } = useTranslation();
  const label = inDomain ? t("applicabilityDomain.inDomain") : t("applicabilityDomain.outOfDomain");
  const tooltip = inDomain
    ? t("applicabilityDomain.inDomainTooltip")
    : t("applicabilityDomain.outOfDomainTooltip");

  return (
    <div
      role="status"
      title={tooltip}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 12px",
        borderRadius: "var(--radius-card)",
        border: `1px solid ${inDomain ? "var(--risk-low)" : "var(--risk-moderate)"}`,
        color: inDomain ? "var(--risk-low)" : "var(--risk-moderate)",
        background: "var(--surface)",
        fontWeight: 600,
        fontSize: 13,
      }}
    >
      <span aria-hidden="true">{inDomain ? "✅" : "⚠️"}</span>
      <span>{label}</span>
    </div>
  );
}
