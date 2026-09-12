import { describe, expect, it } from "vitest";
import {
  COMPARISON_ROWS,
  buildComparisonRadarData,
  buildRadarAxes,
  normalizeSolubility,
  riskLevel,
} from "./admetConstants";
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

describe("COMPARISON_ROWS", () => {
  it("has one row per profile field shown in the dashboard (5 top-level endpoints + 12 Tox21 sub-assays)", () => {
    // Mirrors AdmetProfileDashboard's own layout: 5 top-level cards
    // (including the toxicity_tox21 aggregate) plus the 12-assay Tox21
    // breakdown - together these cover all "16 ADMET tasks" (the
    // aggregate is a derived summary of the 12 sub-assays, not an extra
    // model output), with the aggregate kept as its own row for parity
    // with the single-molecule dashboard.
    expect(COMPARISON_ROWS).toHaveLength(17);
  });

  it("marks risk-style probabilities as 'lower is better' and the rest as 'higher is better'", () => {
    const bySolubility = COMPARISON_ROWS.find((r) => r.key === "solubility");
    const byToxicity = COMPARISON_ROWS.find((r) => r.key === "toxicity_tox21");
    const byAssay = COMPARISON_ROWS.find((r) => r.key === "tox21_NR-AR");

    expect(bySolubility.direction).toBe("higher");
    expect(byToxicity.direction).toBe("lower");
    expect(byAssay.direction).toBe("lower");
    expect(bySolubility.getValue(ASPIRIN_PROFILE.profile)).toBe(-2.31);
    expect(byToxicity.getValue(ASPIRIN_PROFILE.profile)).toBe(0.44);
  });
});

describe("buildComparisonRadarData", () => {
  it("merges each molecule's radar axes into one row per axis, keyed by molecule id", () => {
    const rows = buildComparisonRadarData([
      { id: 0, profile: ASPIRIN_PROFILE.profile },
      { id: 1, profile: ASPIRIN_PROFILE.profile },
    ]);
    expect(rows).toHaveLength(5);
    expect(rows[0]).toHaveProperty("value_0");
    expect(rows[0]).toHaveProperty("value_1");
    expect(rows[0].value_0).toBe(rows[0].value_1);
  });
});
