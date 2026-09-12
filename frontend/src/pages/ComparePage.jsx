import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useQueries } from "@tanstack/react-query";
import { ApiError, predictAdmetProfile } from "../api";
import { COMPARISON_COLORS, COMPARISON_MAX_MOLECULES } from "../admetConstants";
import { useComparisonMolecules } from "../hooks/useComparisonMolecules";
import { SmilesInput } from "../components/SmilesInput";
import { ExampleChips } from "../components/ExampleChips";
import { AdmetComparisonRadarChart } from "../components/AdmetComparisonRadarChart";
import { ComparisonTable } from "../components/ComparisonTable";
import { buildComparisonCsv, buildComparisonRowsForCsv, downloadCsv } from "../csvExport";

function padSlots(list) {
  const slots = list.slice(0, COMPARISON_MAX_MOLECULES);
  while (slots.length < 2) slots.push("");
  return slots;
}

function truncateSmiles(smiles, max = 24) {
  return smiles.length > max ? `${smiles.slice(0, max)}…` : smiles;
}

// Multi-molecule side-by-side comparison - frontend-spec/comparison-page.md,
// TODO_pages_flows.md ("Сторінка порівняння кількох молекул поруч"),
// TODO_data_visualization.md and TODO_state_data_layer.md. Up to 3
// molecules (per the spec's overlaid-radar-legibility cap), fetched in
// parallel (one useQuery per molecule via useQueries) so a slow/failed
// molecule never blocks the others - the same "часткова відмова, не
// блокує все порівняння" state called out in the spec.
export function ComparePage() {
  const { t, i18n } = useTranslation();
  const { molecules: committed, setMoleculeAt, removeMoleculeAt, canAddMore } = useComparisonMolecules();
  const [drafts, setDrafts] = useState(() => padSlots(committed));
  // Re-sync local draft inputs whenever the committed (URL-backed) list
  // changes - e.g. after a submit, a remove, or navigating with the back
  // button to a previously-shared comparison URL. Adjusted directly during
  // render (React's recommended "derive from props" pattern) rather than
  // in an effect, since it's just resetting state in response to a prop-like
  // change, not synchronizing with an external system.
  const [lastCommitted, setLastCommitted] = useState(committed);
  if (committed !== lastCommitted) {
    setLastCommitted(committed);
    setDrafts(padSlots(committed));
  }

  const results = useQueries({
    queries: committed.map((smiles) => ({
      queryKey: ["admet-profile", smiles],
      queryFn: () => predictAdmetProfile(smiles),
      staleTime: Infinity,
      retry: false,
    })),
  });

  const molecules = committed.map((smiles, index) => {
    const result = results[index];
    const status = result?.status === "pending" ? "loading" : result?.status;
    return {
      index,
      smiles,
      label: truncateSmiles(smiles),
      color: COMPARISON_COLORS[index],
      status,
      profile: result?.data?.profile,
      applicabilityDomain: result?.data?.applicability_domain,
      error:
        result?.error instanceof ApiError ? result.error.message : status === "error" ? t("errors.unexpected") : null,
    };
  });

  const successfulMolecules = molecules.filter((m) => m.status === "success" && m.profile);
  const hasEnoughToCompare = committed.length >= 2;

  function handleAddSlot() {
    setDrafts((prev) => (prev.length < COMPARISON_MAX_MOLECULES ? [...prev, ""] : prev));
  }

  function handleRemoveSlot(index) {
    if (index < committed.length) {
      removeMoleculeAt(index);
    } else {
      setDrafts((prev) => prev.filter((_, i) => i !== index));
    }
  }

  function handleExportCsv() {
    const { header, rows } = buildComparisonRowsForCsv(molecules, i18n.language, t);
    const csv = buildComparisonCsv({ header, rows });
    downloadCsv("admet-comparison.csv", csv);
  }

  return (
    <div>
      <h1>{t("compare.title")}</h1>
      <p className="text-muted">{t("compare.subtitle")}</p>

      <div className="card-grid print-hide">
        {drafts.map((draft, index) => (
          <div key={index} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span aria-hidden="true" style={{ color: COMPARISON_COLORS[index], fontSize: 18, lineHeight: 1 }}>
                {"●"}
              </span>
              <span className="text-muted" style={{ fontSize: 12 }}>
                {t("compare.moleculeLabel", { index: index + 1 })}
              </span>
              {index >= 2 && (
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => handleRemoveSlot(index)}
                  aria-label={t("compare.removeMolecule", { index: index + 1 })}
                  style={{ marginLeft: "auto" }}
                >
                  {t("compare.removeButton")}
                </button>
              )}
            </div>
            <SmilesInput
              id={`compare-smiles-input-${index}`}
              value={draft}
              onChange={(value) =>
                setDrafts((prev) => prev.map((s, i) => (i === index ? value : s)))
              }
              onSubmit={() => setMoleculeAt(index, drafts[index])}
              disabled={false}
            />
            {molecules[index]?.status === "loading" && (
              <span role="status" className="text-muted" style={{ fontSize: 12 }}>
                {t("compare.loadingMolecule")}
              </span>
            )}
            {molecules[index]?.status === "error" && (
              <span role="alert" style={{ fontSize: 12, color: "var(--error)" }}>
                <span aria-hidden="true">{"⚠"} </span>
                {molecules[index].error}
              </span>
            )}
          </div>
        ))}
      </div>

      <div className="print-hide" style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
        {canAddMore && drafts.length < COMPARISON_MAX_MOLECULES && (
          <button type="button" className="icon-button" onClick={handleAddSlot}>
            {t("compare.addMolecule")}
          </button>
        )}
        <ExampleChips
          disabled={false}
          onPick={(smiles) => {
            const firstEmpty = drafts.findIndex((d) => !d.trim());
            const index = firstEmpty === -1 ? drafts.length : firstEmpty;
            if (index >= COMPARISON_MAX_MOLECULES) return;
            setDrafts((prev) => {
              const next = [...prev];
              next[index] = smiles;
              return next;
            });
            setMoleculeAt(index, smiles);
          }}
        />
      </div>

      {!hasEnoughToCompare && (
        <p className="text-muted" style={{ marginTop: 24, textAlign: "center" }}>
          {t("compare.needAtLeastTwo")}
        </p>
      )}

      {hasEnoughToCompare && (
        <div style={{ marginTop: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
            <h2 style={{ margin: 0 }}>{t("compare.radarTitle")}</h2>
            <div className="print-hide" style={{ display: "flex", gap: 8 }}>
              <button type="button" className="icon-button" onClick={handleExportCsv}>
                {t("compare.exportCsv")}
              </button>
              <button type="button" className="icon-button" onClick={() => window.print()}>
                {t("compare.exportPrint")}
              </button>
            </div>
          </div>

          {successfulMolecules.length > 0 ? (
            <AdmetComparisonRadarChart molecules={successfulMolecules} />
          ) : (
            <p className="text-muted">{t("compare.noSuccessfulYet")}</p>
          )}

          <h2 style={{ marginTop: 24 }}>{t("compare.tableTitle")}</h2>
          <ComparisonTable molecules={molecules} />

          <p
            role="note"
            className="text-muted"
            style={{ marginTop: 24, fontSize: 12, borderTop: "1px solid var(--border)", paddingTop: 12 }}
          >
            {t("common.disclaimerBanner")}
          </p>
        </div>
      )}
    </div>
  );
}
