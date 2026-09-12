import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictPage } from "./PredictPage";
import { ApiError, predictAdmetProfile } from "../api";
import { renderWithProviders, ASPIRIN_PROFILE } from "../testUtils";

vi.mock("../api", async () => {
  const actual = await vi.importActual("../api");
  return {
    ...actual,
    predictAdmetProfile: vi.fn(),
  };
});

function renderPage() {
  return renderWithProviders(<PredictPage />);
}

beforeEach(() => {
  predictAdmetProfile.mockReset();
});

describe("PredictPage - E2E smoke flow", () => {
  it("goes idle -> loading -> success and shows all 16 tasks plus the disclaimer", async () => {
    const user = userEvent.setup();
    let resolvePredict;
    predictAdmetProfile.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePredict = resolve;
        }),
    );

    renderPage();

    await user.type(screen.getByLabelText("SMILES string"), "CC(=O)Oc1ccccc1C(=O)O");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    expect(screen.getByRole("status")).toHaveTextContent("Computing the 16-task ADMET profile");
    expect(screen.getByLabelText("SMILES string")).toBeDisabled();

    resolvePredict(ASPIRIN_PROFILE);

    await waitFor(() => {
      expect(screen.getByText("Solubility")).toBeInTheDocument();
    });

    // 5 top-level endpoint cards.
    expect(screen.getByText("BBB Penetration")).toBeInTheDocument();
    expect(screen.getByText("Toxicity (Tox21 panel)")).toBeInTheDocument();
    expect(screen.getByText("Clinical Trial Toxicity")).toBeInTheDocument();
    expect(screen.getByText("FDA Approval Likelihood")).toBeInTheDocument();

    // Tox21 12-assay expander is present (collapsed by default).
    expect(screen.getByText(/Toxicity \(Tox21 panel\).*expand 12 assays/)).toBeInTheDocument();

    // Applicability domain badge (in-domain for this fixture).
    expect(screen.getByText("Within known chemical space")).toBeInTheDocument();

    // The disclaimer is always shown alongside the result.
    expect(screen.getByText(ASPIRIN_PROFILE.disclaimer)).toBeInTheDocument();

    expect(screen.queryByText("Computing the 16-task ADMET profile")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the out-of-domain badge and per-card overlay without treating it as an error", async () => {
    const user = userEvent.setup();
    const outOfDomainProfile = {
      ...ASPIRIN_PROFILE,
      applicability_domain: { in_domain: false, distance_score: 4.2 },
    };
    predictAdmetProfile.mockResolvedValueOnce(outOfDomainProfile);

    renderPage();
    await user.type(screen.getByLabelText("SMILES string"), "CCCCCCCCCCCCCCCCCCCC");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() => {
      expect(screen.getByText("Outside training distribution")).toBeInTheDocument();
    });
    // Not an error state - the dashboard still renders.
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("Solubility")).toBeInTheDocument();
    expect(screen.getAllByText(/Low confidence.*outside training data distribution/).length).toBeGreaterThan(0);
  });

  it("goes idle -> loading -> error and shows the error banner for an invalid SMILES, then retries", async () => {
    const user = userEvent.setup();
    let rejectPredict;
    predictAdmetProfile.mockImplementation(
      () =>
        new Promise((_, reject) => {
          rejectPredict = reject;
        }),
    );

    renderPage();
    await user.type(screen.getByLabelText("SMILES string"), "not-a-smiles");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    expect(screen.getByRole("status")).toHaveTextContent("Computing the 16-task ADMET profile");

    rejectPredict(new ApiError("Could not parse this SMILES string", 422));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Could not parse this SMILES string");
    });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    predictAdmetProfile.mockResolvedValueOnce(ASPIRIN_PROFILE);
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => {
      expect(screen.getByText("Solubility")).toBeInTheDocument();
    });
  });

  it("shows a server-unavailable error banner when the backend is unreachable", async () => {
    const user = userEvent.setup();
    predictAdmetProfile.mockRejectedValueOnce(new ApiError("Server unavailable, please try again", 0));

    renderPage();
    await user.type(screen.getByLabelText("SMILES string"), "CCO");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Server unavailable, please try again");
    });
  });
});
