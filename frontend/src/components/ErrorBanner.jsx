import { useTranslation } from "react-i18next";

export function ErrorBanner({ message, onRetry }) {
  const { t } = useTranslation();
  return (
    <div
      role="alert"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "8px 12px",
        borderRadius: "var(--radius-card)",
        background: "var(--surface)",
        border: "1px solid var(--error)",
        color: "var(--error)",
        marginTop: 8,
        flexWrap: "wrap",
        gap: 8,
      }}
    >
      <span>
        <span aria-hidden="true">⚠</span> {message}
      </span>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          style={{
            border: "1px solid var(--error)",
            background: "none",
            color: "var(--error)",
            borderRadius: "var(--radius-chip)",
            padding: "2px 10px",
            cursor: "pointer",
          }}
        >
          {t("errors.retry")}
        </button>
      )}
    </div>
  );
}
