import { describe, it, expect } from "vitest";
import { formatNaira, formatNairaRange } from "../src/lib/format";
import { stripPosition, confidenceFor, statusFor } from "../src/lib/stats";

describe("naira formatting", () => {
  it("formats annual values", () => {
    expect(formatNaira(1500000)).toContain("NGN");
    expect(formatNaira(null)).toBe("No data available");
    expect(formatNairaRange(1000000, 2000000)).toContain("to");
  });
});

describe("range strip scaling", () => {
  it("scales min median max", () => {
    const p = stripPosition(1000000, 1500000, 2000000, 0, 2000000);
    expect(p.left).toBe(50);
    expect(p.width).toBe(50);
    expect(p.marker).toBe(75);
  });
});

describe("threshold states", () => {
  it("maps counts to confidence and status", () => {
    expect(statusFor(0)).toBe("no_data");
    expect(statusFor(2)).toBe("low_data");
    expect(confidenceFor(2)).toBe("insufficient");
    expect(confidenceFor(5)).toBe("limited");
    expect(confidenceFor(12)).toBe("normal");
  });
});
