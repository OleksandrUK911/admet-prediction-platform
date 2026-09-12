import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ApiError, getHistory } from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import { formatDate } from "../formatters";

function truncateSmiles(smiles, max = 32) {
  return smiles.length > max ? `${smiles.slice(0, max)}…` : smiles;
}

// GET /history - list of previously computed profiles, newest first,
// persisted server-side (the frontend never writes history itself). Each
// row links to /profile/:id (GET /admet-profile/{id}) rather than
// re-running the prediction - frontend-spec README + TODO_pages_flows.md.
export function HistoryPage() {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();

  const { status, data, error, refetch } = useQuery({
    queryKey: ["history"],
    queryFn: getHistory,
  });
  const rows = data ?? [];

  if (status === "pending") {
    return (
      <div>
        <h1>{t("history.title")}</h1>
        <p className="text-muted">{t("history.loading")}</p>
      </div>
    );
  }

  if (status === "error") {
    const message = error instanceof ApiError ? error.message : t("errors.unexpected");
    return (
      <div>
        <h1>{t("history.title")}</h1>
        <ErrorBanner message={message} onRetry={refetch} />
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div style={{ textAlign: "center", marginTop: 48 }}>
        <h1>{t("history.title")}</h1>
        <p className="text-muted">{t("history.empty")}</p>
        <Link to="/" style={{ color: "var(--accent)" }}>
          {t("history.firstPrediction")}
        </Link>
      </div>
    );
  }

  return (
    <div>
      <h1>{t("history.title")}</h1>
      <div className="table-scroll">
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 480 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <th style={{ textAlign: "left", padding: "8px 4px" }}>{t("history.columnSmiles")}</th>
              <th style={{ textAlign: "left", padding: "8px 4px" }}>{t("history.columnDate")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} style={{ borderBottom: "1px solid var(--border)" }}>
                <td title={row.smiles} style={{ padding: "8px 4px" }}>
                  {truncateSmiles(row.smiles)}
                </td>
                <td style={{ padding: "8px 4px" }} className="text-muted">
                  {formatDate(row.created_at, i18n.language)}
                </td>
                <td style={{ padding: "8px 4px" }}>
                  <button
                    type="button"
                    onClick={() => navigate(`/profile/${row.id}`)}
                    style={{ background: "none", border: "none", color: "var(--accent)", cursor: "pointer" }}
                  >
                    {t("history.viewButton")}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
