import { useState } from "react";
import { useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ApiError, predictAdmetProfile } from "../api";
import { SmilesInput } from "../components/SmilesInput";
import { ExampleChips } from "../components/ExampleChips";
import { ErrorBanner } from "../components/ErrorBanner";
import { AdmetProfileDashboard } from "../components/AdmetProfileDashboard";

// Home / predict page - frontend-spec/predict-page.md. Out-of-domain
// results are NOT an error state: the full dashboard still renders, just
// with the low-confidence overlay/badge (see AdmetProfileDashboard) -
// this is a deliberate UX distinction from an actually-invalid SMILES or
// server error, called out in TODO_pages_flows.md.
export function PredictPage() {
  const { t } = useTranslation();
  const location = useLocation();
  const queryClient = useQueryClient();
  // Came from History page's "View" action - pre-fill, let the user
  // re-submit (we don't cache full prediction detail in history rows).
  const [smiles, setSmiles] = useState(location.state?.smiles || "");

  // Same pattern as project #1: useMutation drives the loading/error UX,
  // but the mutationFn routes through the query cache keyed by SMILES so
  // repeat submissions of the same molecule are served from cache instead
  // of re-hitting the (potentially slow, multi-task) API - TODO_state_data_layer.md.
  const mutation = useMutation({
    mutationFn: (inputSmiles) =>
      queryClient.fetchQuery({
        queryKey: ["admet-profile", inputSmiles],
        queryFn: () => predictAdmetProfile(inputSmiles),
        staleTime: Infinity,
      }),
  });

  function runPrediction(inputSmiles) {
    mutation.mutate(inputSmiles);
  }

  const status = mutation.status === "pending" ? "loading" : mutation.status;
  const result = mutation.data;
  const error = mutation.error instanceof ApiError ? mutation.error.message : t("errors.unexpected");

  return (
    <div>
      <h1>{t("predict.title")}</h1>
      <SmilesInput
        value={smiles}
        onChange={setSmiles}
        onSubmit={() => runPrediction(smiles)}
        disabled={status === "loading"}
      />
      <ExampleChips
        disabled={status === "loading"}
        onPick={(s) => {
          setSmiles(s);
          runPrediction(s);
        }}
      />

      {status === "error" && <ErrorBanner message={error} onRetry={() => runPrediction(smiles)} />}

      {status === "loading" && (
        <div style={{ marginTop: 24 }} role="status" aria-live="polite">
          <p className="text-muted" style={{ textAlign: "center" }}>
            {t("predict.predicting")}
          </p>
          <div className="card-grid">
            {Array.from({ length: 5 }).map((_, i) => (
              // eslint-disable-next-line react/no-array-index-key -- static skeleton count, order never changes
              <div
                key={i}
                aria-hidden="true"
                style={{
                  height: 96,
                  borderRadius: "var(--radius-card)",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                }}
              />
            ))}
          </div>
        </div>
      )}

      {status === "success" && result && (
        <div style={{ marginTop: 24 }}>
          <AdmetProfileDashboard
            profile={result.profile}
            applicabilityDomain={result.applicability_domain}
            smiles={result.smiles}
            disclaimer={result.disclaimer}
          />
        </div>
      )}
    </div>
  );
}
