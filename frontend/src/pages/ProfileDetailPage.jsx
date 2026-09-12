import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ApiError, getAdmetProfile } from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import { AdmetProfileDashboard } from "../components/AdmetProfileDashboard";

// Shareable/saved-profile view - GET /admet-profile/{id}, linked to from
// the History page's "View" action (backend-spec/api-contract.md).
// Renders the exact same dashboard as a freshly-submitted PredictPage
// result, since the response shape is identical either way.
export function ProfileDetailPage() {
  const { id } = useParams();
  const { t } = useTranslation();

  const { status, data, error, refetch } = useQuery({
    queryKey: ["admet-profile", "by-id", id],
    queryFn: () => getAdmetProfile(id),
  });

  if (status === "pending") {
    return (
      <div>
        <h1>{t("profileDetail.title")}</h1>
        <p className="text-muted">{t("profileDetail.loading")}</p>
      </div>
    );
  }

  if (status === "error") {
    const message = error instanceof ApiError ? error.message : t("errors.unexpected");
    return (
      <div>
        <h1>{t("profileDetail.title")}</h1>
        <ErrorBanner message={message} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div>
      <h1>{t("profileDetail.title")}</h1>
      <AdmetProfileDashboard
        profile={data.profile}
        applicabilityDomain={data.applicability_domain}
        smiles={data.smiles}
        disclaimer={data.disclaimer}
      />
    </div>
  );
}
