import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Shared test helper: components under test use React Query hooks
// (useQuery/useMutation), which require a QueryClientProvider ancestor.
// A fresh QueryClient per render keeps each test's cache isolated.
export function renderWithProviders(ui, { route = "/", initialEntries } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries ?? [route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

export const ASPIRIN_PROFILE = {
  id: "b3f1c2a0-0000-0000-0000-000000000000",
  smiles: "CC(=O)Oc1ccccc1C(=O)O",
  disclaimer: "Research/educational use only. Not for clinical or regulatory decision-making.",
  model_version: "0.1.0",
  applicability_domain: { in_domain: true, distance_score: 0.31 },
  profile: {
    solubility: { value: -2.31, unit: "logS", confidence: 0.82 },
    bbb_penetration: { probability: 0.74, confidence: 0.68 },
    toxicity_tox21: {
      aggregate_risk: 0.44,
      confidence: 0.71,
      panel: {
        "NR-AR": 0.05,
        "NR-AR-LBD": 0.03,
        "NR-AhR": 0.41,
        "NR-Aromatase": 0.08,
        "NR-ER": 0.12,
        "NR-ER-LBD": 0.09,
        "NR-PPAR-gamma": 0.02,
        "SR-ARE": 0.33,
        "SR-ATAD5": 0.06,
        "SR-HSE": 0.11,
        "SR-MMP": 0.44,
        "SR-p53": 0.07,
      },
    },
    clinical_trial_toxicity: { probability: 0.15, confidence: 0.65 },
    fda_approval_likelihood: { probability: 0.58, confidence: 0.6 },
  },
};
