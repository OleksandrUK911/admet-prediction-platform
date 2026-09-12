import { describe, expect, it } from "vitest";
import { buildRadarAxes, normalizeSolubility, riskLevel } from "./admetConstants";
import { ASPIRIN_PROFILE } from "./testUtils.jsx";

describe("riskLevel", () => {
  it("classifies below 0.30 as low", () => {
    expect(riskLevel(0.0)).toBe("low");
    expect(riskLevel(0.29)).toBe("low");
  });

  it("classifies 0.30-0.60 as moderate", () => {
    expect(riskLevel(0.3)).toBe("moderate");
    expect(riskLevel(0.59)).toBe("moderate");
  });

  it("classifies 0.60 and above as high", () => {
    expect(riskLevel(0.6)).toBe("high");
    expect(riskLevel(1.0)).toBe("high");
  });
});

describe("normalizeSolubility", () => {
  it("clamps to [0, 1]", () => {
    expect(normalizeSolubility(-8)).toBe(0);
    expect(normalizeSolubility(2)).toBe(1);
    expect(normalizeSolubility(-20)).toBe(0);
    expect(normalizeSolubility(20)).toBe(1);
  });
});

describe("buildRadarAxes", () => {
  it("produces 5 axes and inverts the two toxicity axes so higher = safer", () => {
    const axes = buildRadarAxes(ASPIRIN_PROFILE.profile);
    expect(axes).toHaveLength(5);

    const toxicity = axes.find((a) => a.key === "toxicity_tox21");
    expect(toxicity.rawValue).toBe(0.44);
    expect(toxicity.radarValue).toBeCloseTo(1 - 0.44);
    expect(toxicity.inverted).toBe(true);

    const bbb = axes.find((a) => a.key === "bbb_penetration");
    expect(bbb.radarValue).toBe(bbb.rawValue);
    expect(bbb.inverted).toBe(false);
  });
});
