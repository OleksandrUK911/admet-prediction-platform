import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AdmetResultCard } from "./AdmetResultCard";

describe("AdmetResultCard", () => {
  it("renders the value and a confidence badge/bar when confidence is provided", () => {
    render(
      <AdmetResultCard
        titleKey="tasks.bbbPenetration"
        valueLabel="74%"
        probability={0.74}
        confidence={0.68}
        outOfDomain={false}
      />,
    );
    expect(screen.getByText("BBB Penetration")).toBeInTheDocument();
    expect(screen.getByText("74%")).toBeInTheDocument();
    expect(screen.getByText("68%")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Confidence: 68%/ })).toBeInTheDocument();
  });

  it("shows a risk-level badge with text (not color alone) at high risk", () => {
    render(
      <AdmetResultCard
        titleKey="tasks.toxicityTox21"
        valueLabel="82%"
        probability={0.82}
        confidence={0.5}
        outOfDomain={false}
      />,
    );
    expect(screen.getByText("High risk")).toBeInTheDocument();
  });

  it("shows a low-risk badge for a low probability", () => {
    render(
      <AdmetResultCard
        titleKey="tasks.clinicalTrialToxicity"
        valueLabel="10%"
        probability={0.1}
        confidence={0.9}
        outOfDomain={false}
      />,
    );
    expect(screen.getByText("Low risk")).toBeInTheDocument();
  });

  it("renders no risk badge for a non-probability value (e.g. solubility)", () => {
    render(
      <AdmetResultCard
        titleKey="tasks.solubility"
        valueLabel="-2.31 logS"
        probability={null}
        confidence={0.82}
        outOfDomain={false}
      />,
    );
    expect(screen.queryByText(/risk/i)).not.toBeInTheDocument();
  });

  it("renders the no-confidence note when confidence is null", () => {
    render(
      <AdmetResultCard titleKey="tasks.solubility" valueLabel="-2.31 logS" probability={null} confidence={null} outOfDomain={false} />,
    );
    expect(screen.getByText(/No calibrated confidence measure exists/)).toBeInTheDocument();
  });

  it("renders the out-of-domain overlay when applicable", () => {
    render(
      <AdmetResultCard
        titleKey="tasks.bbbPenetration"
        valueLabel="74%"
        probability={0.74}
        confidence={0.68}
        outOfDomain={true}
      />,
    );
    expect(screen.getByText(/Low confidence.*outside training data distribution/)).toBeInTheDocument();
  });
});
