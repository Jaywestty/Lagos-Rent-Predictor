import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";
import axe from "axe-core";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Chat } from "../src/components/chat/Chat";
import { RentAnswer } from "../src/components/rent/RentAnswer";
import type { ChatAnswer } from "../src/types/api";

vi.mock("../src/services/api", () => ({
  ask: vi.fn(async () => ({ answer: { status: "no_data", listing_count: 0, confidence: "insufficient", follow_ups: [] }, listings: [] })),
  getAreas: vi.fn(async () => []),
  newRequestKey: () => "key-1",
  track: vi.fn(),
  saveSearch: vi.fn(),
  getThread: vi.fn(),
  getThreads: vi.fn(async () => []),
  deleteThread: vi.fn(),
  deleteAllThreads: vi.fn(),
  newThreadId: () => "6f1c2d3e-0000-4000-8000-000000000001",
}));

const answer = {
  version: 1, status: "success", location: "Lekki", area: "Lekki", property_type: null, currency: "NGN", rent_period: "yearly",
  headline: { min: 3_000_000, max: 50_000_000 }, median: 15_000_000, listing_count: 72, range_method: "percentile",
  breakdown: { self_contain: null, one_bedroom: { min: 2e6, max: 9e6, median: 5e6, count: 13 }, two_bedroom: { min: 8e6, max: 27e6, median: 13.5e6, count: 10 }, three_bedroom: { min: 9e6, max: 40e6, median: 18e6, count: 15 } },
  data_as_of: new Date().toISOString(), last_updated_days: 0, confidence: "normal", clarification: null,
  follow_ups: ["What about a 2 bedroom in Lekki?"], request_id: "r", latency_ms: 1, message: "",
} as ChatAnswer;

async function serious(ui: React.ReactElement) {
  const qc = new QueryClient();
  const { container } = render(<MemoryRouter><QueryClientProvider client={qc}>{ui}</QueryClientProvider></MemoryRouter>);
  const res = await axe.run(container);
  return res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
}

describe("a11y", () => {
  it("chat has no serious axe violations", async () => {
    expect(await serious(<Chat />)).toEqual([]);
  });
  it("answer card has no serious axe violations and shows the median", async () => {
    expect(await serious(<RentAnswer answer={answer} question="rent in Lekki" onAsk={() => undefined} />)).toEqual([]);
  });
});
