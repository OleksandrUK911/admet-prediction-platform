import { useTranslation } from "react-i18next";

export function SmilesInput({ value, onChange, onSubmit, disabled }) {
  const { t } = useTranslation();
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      style={{ display: "flex", gap: 8, flexWrap: "wrap" }}
    >
      <label htmlFor="smiles-input" className="visually-hidden">
        {t("predict.smilesLabel")}
      </label>
      <input
        id="smiles-input"
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
        aria-label={t("predict.smilesLabel")}
        disabled={disabled}
        style={{
          flex: 1,
          minWidth: 200,
          padding: "8px 12px",
          borderRadius: "var(--radius-card)",
          border: "1px solid var(--border)",
          background: "var(--surface)",
          color: "var(--text)",
        }}
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        style={{
          padding: "8px 20px",
          borderRadius: "var(--radius-card)",
          border: "none",
          background: disabled || !value.trim() ? "var(--text-muted)" : "var(--accent-hover)",
          color: "white",
          cursor: disabled || !value.trim() ? "not-allowed" : "pointer",
        }}
      >
        {t("predict.submitButton")}
      </button>
    </form>
  );
}
