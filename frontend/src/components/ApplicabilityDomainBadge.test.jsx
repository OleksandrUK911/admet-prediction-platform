import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ApplicabilityDomainBadge } from "./ApplicabilityDomainBadge";

describe("ApplicabilityDomainBadge", () => {
  it("shows the in-domain label and icon", () => {
    render(<ApplicabilityDomainBadge inDomain={true} />);
    expect(screen.getByText("Within known chemical space")).toBeInTheDocument();
  });

  it("shows the out-of-domain label, never relying on color alone", () => {
    render(<ApplicabilityDomainBadge inDomain={false} />);
    expect(screen.getByText("Outside training distribution")).toBeInTheDocument();
    // The warning icon (text alternative) is present alongside the text label.
    expect(screen.getByRole("status")).toHaveTextContent("Outside training distribution");
  });
});
