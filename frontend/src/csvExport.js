// Client-side CSV export for the comparison table - TODO_pages_flows.md
// ("Флоу експорту результатів у PDF/CSV") + TODO_testing.md. No backend
// endpoint: the comparison data is already fetched into the page, so this
// just serializes it and triggers a browser download. PDF export is not
// implemented here - see ComparePage's print stylesheet for the pragmatic
// substitute (window.print() to "Save as PDF"), documented in
// TODO_pages_flows.md rather than a jsPDF dependency.
import { COMPARISON_ROWS } from "./admetConstants";
import { formatNumber, formatPercent } from "./formatters";

function formatValueForCsv(row, value, unit, language) {
  if (value === null || value === undefined) return "";
  if (row.kind === "probability") return formatPercent(value, language);
  return `${formatNumber(value, language)}${unit ? ` ${unit}` : ""}`;
}

// Builds the same rows shown in <ComparisonTable> as a plain
// header+rows structure ready for buildComparisonCsv. `molecules` is the
// ComparePage shape: [{ smiles, status, profile }].
export function buildComparisonRowsForCsv(molecules, language, t) {
  const header = [t("compare.columnEndpoint"), ...molecules.map((m) => m.smiles), t("compare.columnLargestDiff")];
  const rows = COMPARISON_ROWS.map((row) => {
    const values = molecules.map((m) => (m.status === "success" ? row.getValue(m.profile) : null));
    const valid = values.filter((v) => v !== null && v !== undefined);
    const spread = valid.length >= 2 ? Math.max(...valid) - Math.min(...valid) : null;
    const successfulProfile = molecules.find((m) => m.status === "success")?.profile;
    const unit = row.kind === "regression" && successfulProfile ? row.getUnit(successfulProfile) : null;
    const label = row.titleKey ? t(row.titleKey) : row.label;
    return [
      label,
      ...molecules.map((m, i) => (m.status === "success" ? formatValueForCsv(row, values[i], unit, language) : m.status === "error" ? "ERROR" : "")),
      spread !== null ? formatValueForCsv(row, spread, unit, language) : "",
    ];
  });
  return { header, rows };
}

function escapeCsvCell(value) {
  const str = String(value ?? "");
  if (/[",\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function buildComparisonCsv({ header, rows }) {
  const lines = [header, ...rows].map((row) => row.map(escapeCsvCell).join(","));
  // Leading BOM so Excel opens UTF-8 (Cyrillic labels/units) correctly.
  return `﻿${lines.join("\r\n")}`;
}

export function downloadCsv(filename, csvContent) {
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
