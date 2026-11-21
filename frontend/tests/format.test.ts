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

import { nairaShort, timeAgo } from "../src/lib/format";

describe("short naira and relative time", () => {
  it("shortens naira", () => {
    expect(nairaShort(15_000_000)).toBe("₦15M");
    expect(nairaShort(1_500_000)).toBe("₦1.5M");
    expect(nairaShort(850_000)).toBe("₦850k");
    expect(nairaShort(null)).toBe("-");
  });
  it("describes listing age", () => {
    const now = new Date("2026-11-21T12:00:00Z");
    expect(timeAgo("2026-11-21T08:00:00Z", now)).toBe("today");
    expect(timeAgo("2026-11-20T08:00:00Z", now)).toBe("yesterday");
    expect(timeAgo("2026-11-16T08:00:00Z", now)).toBe("5 days ago");
    expect(timeAgo(null, now)).toBe("date unknown");
  });
});
