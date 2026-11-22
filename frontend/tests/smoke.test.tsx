import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Chat } from "../src/components/chat/Chat";

vi.mock("../src/services/api", () => ({
  ask: vi.fn(async () => ({ answer: { status: "no_data", listing_count: 0, confidence: "insufficient", follow_ups: [] }, listings: [] })),
  getAreas: vi.fn(async () => [{ area: "Lekki", listing_count: 72, median_ngn: 15000000, latest_listing_at: null }]),
  newRequestKey: () => "key-1",
  track: vi.fn(),
  saveSearch: vi.fn(),
  getThread: vi.fn(),
  getThreads: vi.fn(async () => []),
  deleteThread: vi.fn(),
  deleteAllThreads: vi.fn(),
  newThreadId: () => "6f1c2d3e-0000-4000-8000-000000000001",
}));

function wrap() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <MemoryRouter><QueryClientProvider client={qc}><Chat /></QueryClientProvider></MemoryRouter>;
}

describe("smoke", () => {
  it("renders landing at 360 width", () => {
    Object.defineProperty(window, "innerWidth", { value: 360, configurable: true });
    render(wrap());
    expect(screen.getByText(/Rent in Lagos/i)).toBeTruthy();
    expect(screen.getByPlaceholderText(/How much is rent in Gbagada/i)).toBeTruthy();
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(360 + 2);
  });
  it("renders at desktop width with quick-search chips", async () => {
    Object.defineProperty(window, "innerWidth", { value: 1280, configurable: true });
    render(wrap());
    expect(screen.getByText(/Ask like you talk/i)).toBeTruthy();
    expect(await screen.findByRole("button", { name: "Lekki" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "2 bed" })).toBeTruthy();
  });
  it("retry preserves question", async () => {
    render(wrap());
    const input = screen.getByLabelText("Your question") as HTMLInputElement;
    input.value = "rent in Yaba";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    expect(input.value).toBe("rent in Yaba");
  });
});
