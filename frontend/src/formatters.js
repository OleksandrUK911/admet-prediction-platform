// Shared locale-aware formatting helpers - see i18n.js for supported
// languages. Centralizing this keeps number/date formatting consistent
// (and actually locale-aware) instead of scattering plain .toFixed(2) /
// .toLocaleString() calls that ignore the app's current language.

export function formatNumber(value, language, options = {}) {
  if (typeof value !== "number" || Number.isNaN(value)) return value;
  return new Intl.NumberFormat(language, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    ...options,
  }).format(value);
}

export function formatPercent(value, language) {
  if (typeof value !== "number" || Number.isNaN(value)) return value;
  return new Intl.NumberFormat(language, {
    style: "percent",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatDate(value, language) {
  const date = value instanceof Date ? value : new Date(value);
  return new Intl.DateTimeFormat(language, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}
