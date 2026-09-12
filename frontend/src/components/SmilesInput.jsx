import { useTranslation } from "react-i18next";

// `id` defaults to "smiles-input" for the single-instance case
// (PredictPage); the comparison page renders several instances at once
// and must pass a unique id per slot to keep label association valid
// (duplicate DOM ids break getByLabelText/assistive tech) -
// TODO_state_data_layer.md / comparison-page.md.
export function SmilesInput({ value, onChange, onSubmit, disabled, id = "smiles-input" }) {
  const { t } = useTranslation();
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      style={{ display: "flex", gap: 8, flexWrap: "wrap" }}
    >
      <label htmlFor={id} className="visually-hidden">
        {t("predict.smilesLabel")}
      </label>
      <input
        id={id}
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
