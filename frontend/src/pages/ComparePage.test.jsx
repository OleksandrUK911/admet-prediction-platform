import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ComparePage } from "./ComparePage";
import { ApiError, predictAdmetProfile } from "../api";
import { renderWithProviders, ASPIRIN_PROFILE } from "../testUtils";

vi.mock("../api", async () => {
  const actual = await vi.importActual("../api");
  return {
    ...actual,
    predictAdmetProfile: vi.fn(),
  };
});

const CAFFEINE_PROFILE = {
  ...ASPIRIN_PROFILE,
  smiles: "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  profile: {
    ...ASPIRIN_PROFILE.profile,
    solubility: { value: -0.5, unit: "logS", confidence: 0.9 },
    toxicity_tox21: { ...ASPIRIN_PROFILE.profile.toxicity_tox21, aggregate_risk: 0.1 },
  },
};

function renderPage(route) {
  return renderWithProviders(<ComparePage />, { route: route ?? "/compare" });
}

beforeEach(() => {
  predictAdmetProfile.mockReset();
});

describe("ComparePage", () => {
  it("shows a placeholder until at least 2 molecules are entered", () => {
    renderPage();
    expect(screen.getByText(/Add at least 2 molecules to compare/)).toBeInTheDocument();
    // Two empty SMILES slots are shown by default.
    expect(screen.getAllByLabelText("SMILES string")).toHaveLength(2);
  });

  it("fetches each molecule in parallel and renders the overlaid radar plus the 16-task table", async () => {
    const user = userEvent.setup();
    predictAdmetProfile.mockImplementation((smiles) =>
      Promise.resolve(smiles === CAFFEINE_PROFILE.smiles ? CAFFEINE_PROFILE : ASPIRIN_PROFILE),
    );

    renderPage();
    const inputs = screen.getAllByLabelText("SMILES string");
    await user.type(inputs[0], ASPIRIN_PROFILE.smiles);
    await user.keyboard("{Enter}");
    await user.type(inputs[1], CAFFEINE_PROFILE.smiles);
    await user.keyboard("{Enter}");

    expect(predictAdmetProfile).toHaveBeenCalledWith(ASPIRIN_PROFILE.smiles);
    expect(predictAdmetProfile).toHaveBeenCalledWith(CAFFEINE_PROFILE.smiles);

    await waitFor(() => {
      expect(screen.getByText("Comparison table (16 tasks)")).toBeInTheDocument();
    });
    // Table has a row per task, including Tox21 sub-assays.
    expect(screen.getByText("NR-AR — Androgen receptor activity")).toBeInTheDocument();
    // Best-value marker (check icon, not color alone) appears somewhere.
    expect(screen.getAllByText("Best value in this row").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Worst value in this row").length).toBeGreaterThan(0);
  });

  it("shows a partial failure without blocking the rest of the comparison", async () => {
    const user = userEvent.setup();
    predictAdmetProfile.mockImplementation((smiles) =>
      smiles === "bad-smiles"
        ? Promise.reject(new ApiError("Could not parse this SMILES string", 422))
        : Promise.resolve(ASPIRIN_PROFILE),
    );

    renderPage();
    const inputs = screen.getAllByLabelText("SMILES string");
    await user.type(inputs[0], ASPIRIN_PROFILE.smiles);
    await user.keyboard("{Enter}");
    await user.type(inputs[1], "bad-smiles");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Could not parse this SMILES string")).toBeInTheDocument();
    });
    // The comparison table still renders for the successful molecule.
    expect(screen.getByText("Comparison table (16 tasks)")).toBeInTheDocument();
    expect(screen.getAllByText("Could not be computed for this molecule").length).toBeGreaterThan(0);
  });

  it("initializes molecules from the URL query string (shareable/bookmarkable)", async () => {
    predictAdmetProfile.mockResolvedValue(ASPIRIN_PROFILE);
    renderPage(`/compare?m=${encodeURIComponent(ASPIRIN_PROFILE.smiles)}&m=${encodeURIComponent(CAFFEINE_PROFILE.smiles)}`);

    await waitFor(() => {
      expect(predictAdmetProfile).toHaveBeenCalledWith(ASPIRIN_PROFILE.smiles);
      expect(predictAdmetProfile).toHaveBeenCalledWith(CAFFEINE_PROFILE.smiles);
    });
    await waitFor(() => {
      expect(screen.getByText("Comparison table (16 tasks)")).toBeInTheDocument();
    });
  });

  it("supports adding a 3rd molecule and removing it again", async () => {
    const user = userEvent.setup();
    renderPage();
    expect(screen.getAllByLabelText("SMILES string")).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "+ Add molecule" }));
    expect(screen.getAllByLabelText("SMILES string")).toHaveLength(3);
    expect(screen.queryByRole("button", { name: "+ Add molecule" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Remove molecule 3" }));
    expect(screen.getAllByLabelText("SMILES string")).toHaveLength(2);
  });
});
