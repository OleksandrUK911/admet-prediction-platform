import { describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { HistoryPage } from "./HistoryPage";
import { ApiError, getHistory } from "../api";
import { renderWithProviders } from "../testUtils";

vi.mock("../api", async () => {
  const actual = await vi.importActual("../api");
  return {
    ...actual,
    getHistory: vi.fn(),
  };
});

function makeRow(overrides = {}) {
  return {
    id: "abc-123",
    smiles: "CC(=O)Oc1ccccc1C(=O)O",
    created_at: "2024-01-01T10:00:00Z",
    ...overrides,
  };
}

describe("HistoryPage", () => {
  it("shows an empty state with a link to make a first prediction", async () => {
    getHistory.mockResolvedValueOnce([]);
    renderWithProviders(<HistoryPage />);

    await waitFor(() => {
      expect(screen.getByText("No predictions yet.")).toBeInTheDocument();
    });
    expect(screen.getByRole("link", { name: "Make your first prediction" })).toBeInTheDocument();
  });

  it("lists history rows with a View link to the saved profile", async () => {
    getHistory.mockResolvedValueOnce([makeRow()]);
    renderWithProviders(<HistoryPage />);

    await waitFor(() => {
      expect(screen.getByText(/CC\(=O\)Oc1ccccc1C\(=O\)O/)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: "View" })).toBeInTheDocument();
  });

  it("shows a retryable error banner when the backend is unavailable", async () => {
    getHistory.mockRejectedValueOnce(new ApiError("Server unavailable, please try again", 0));
    renderWithProviders(<HistoryPage />);

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Server unavailable, please try again");
    });
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
