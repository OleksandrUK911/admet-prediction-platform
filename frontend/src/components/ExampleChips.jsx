import { useTranslation } from "react-i18next";
import { EXAMPLE_MOLECULES } from "../admetConstants";

export function ExampleChips({ onPick, disabled }) {
  const { t } = useTranslation();
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap", alignItems: "center" }}>
      <span className="text-muted">{t("predict.examplesLabel")}</span>
      {EXAMPLE_MOLECULES.map((ex) => (
        <button
          key={ex.label}
          type="button"
          disabled={disabled}
          onClick={() => onPick(ex.smiles)}
          style={{
            padding: "4px 10px",
            borderRadius: "var(--radius-chip)",
            border: "1px solid var(--border)",
            background: "var(--surface)",
            color: "var(--text)",
            cursor: disabled ? "not-allowed" : "pointer",
          }}
        >
          {ex.label}
        </button>
      ))}
    </div>
  );
}
