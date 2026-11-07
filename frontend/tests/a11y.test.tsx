import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";
import axe from "axe-core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Chat } from "../src/components/chat/Chat";

vi.mock("../src/services/api", () => ({
  ask: vi.fn(async () => ({ status: "no_data", listing_count: 0, confidence: "insufficient", follow_ups: [] })),
  newRequestKey: () => "key-1",
}));

describe("a11y", () => {
  it("no serious axe violations", async () => {
    const qc = new QueryClient();
    const { container } = render(<QueryClientProvider client={qc}><Chat /></QueryClientProvider>);
    const res = await axe.run(container);
    const serious = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious).toEqual([]);
  });
});
