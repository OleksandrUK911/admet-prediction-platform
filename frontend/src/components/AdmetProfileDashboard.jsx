import { useTranslation } from "react-i18next";
import { AdmetRadarChart } from "./AdmetRadarChart";
import { AdmetResultCard } from "./AdmetResultCard";
import { ApplicabilityDomainBadge } from "./ApplicabilityDomainBadge";
import { Tox21PanelChart } from "./Tox21PanelChart";
import { formatNumber } from "../formatters";

// Full 16-task ADMET profile dashboard - shared between PredictPage (a
// freshly-computed profile) and ProfileDetailPage (a saved profile loaded
// by id), so both paths render identically. Layout per
// frontend-spec/predict-page.md: radar + applicability-domain badge side
// by side, then 5 endpoint cards, then the Tox21 12-assay expander.
export function AdmetProfileDashboard({ profile, applicabilityDomain, smiles, disclaimer }) {
  const { t, i18n } = useTranslation();
  const outOfDomain = !applicabilityDomain.in_domain;

  return (
    <div>
      <p className="text-muted" style={{ wordBreak: "break-all" }}>
        <strong>{t("results.smilesLabel")}:</strong> {smiles}
      </p>

      <div className="two-column">
        <div>
          <h2>{t("results.radarTitle")}</h2>
          <AdmetRadarChart profile={profile} />
        </div>
        <div>
          <h2>{t("results.applicabilityDomainTitle")}</h2>
          <ApplicabilityDomainBadge inDomain={applicabilityDomain.in_domain} />
          {outOfDomain && (
            <p className="text-muted" style={{ fontSize: 12, marginTop: 8 }}>
              {t("applicabilityDomain.outOfDomainExplanation")}
            </p>
          )}
        </div>
      </div>

      <h2 style={{ marginTop: 24 }}>{t("results.profileTitle")}</h2>
      <div className="card-grid">
        <AdmetResultCard
          titleKey="tasks.solubility"
          valueLabel={`${formatNumber(profile.solubility.value, i18n.language)} ${profile.solubility.unit}`}
          probability={null}
          confidence={profile.solubility.confidence}
          outOfDomain={outOfDomain}
        />
        <AdmetResultCard
          titleKey="tasks.bbbPenetration"
          valueLabel={`${Math.round(profile.bbb_penetration.probability * 100)}%`}
          probability={profile.bbb_penetration.probability}
          confidence={profile.bbb_penetration.confidence}
          outOfDomain={outOfDomain}
        />
        <AdmetResultCard
          titleKey="tasks.toxicityTox21"
          valueLabel={`${Math.round(profile.toxicity_tox21.aggregate_risk * 100)}%`}
          probability={profile.toxicity_tox21.aggregate_risk}
          confidence={profile.toxicity_tox21.confidence}
          outOfDomain={outOfDomain}
        />
        <AdmetResultCard
          titleKey="tasks.clinicalTrialToxicity"
          valueLabel={`${Math.round(profile.clinical_trial_toxicity.probability * 100)}%`}
          probability={profile.clinical_trial_toxicity.probability}
          confidence={profile.clinical_trial_toxicity.confidence}
          outOfDomain={outOfDomain}
        />
        <AdmetResultCard
          titleKey="tasks.fdaApprovalLikelihood"
          valueLabel={`${Math.round(profile.fda_approval_likelihood.probability * 100)}%`}
          probability={profile.fda_approval_likelihood.probability}
          confidence={profile.fda_approval_likelihood.confidence}
          outOfDomain={outOfDomain}
        />
      </div>

      <Tox21PanelChart panel={profile.toxicity_tox21.panel} confidence={profile.toxicity_tox21.confidence} />

      <p
        role="note"
        className="text-muted"
        style={{ marginTop: 24, fontSize: 12, borderTop: "1px solid var(--border)", paddingTop: 12 }}
      >
        {disclaimer}
      </p>
    </div>
  );
}
