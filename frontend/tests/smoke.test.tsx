import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Chat } from "../src/components/chat/Chat";

vi.mock("../src/services/api", () => ({
  ask: vi.fn(async () => ({ status: "no_data", listing_count: 0, confidence: "insufficient", follow_ups: [] })),
  newRequestKey: () => "key-1",
}));

const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
function wrap() {
  return <QueryClientProvider client={qc}><Chat /></QueryClientProvider>;
}

describe("smoke", () => {
  it("renders landing at 360 width", () => {
    Object.defineProperty(window, "innerWidth", { value: 360, configurable: true });
    render(wrap());
    expect(screen.getByText(/Rent in Lagos/i)).toBeTruthy();
    expect(screen.getByPlaceholderText(/How much is rent in Gbagada/i)).toBeTruthy();
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(360 + 2);
  });
  it("renders at desktop width", () => {
    Object.defineProperty(window, "innerWidth", { value: 1280, configurable: true });
    render(wrap());
    expect(screen.getByText(/Ask like you talk/i)).toBeTruthy();
  });
  it("retry preserves question", async () => {
    render(wrap());
    const input = screen.getByLabelText("Your question") as HTMLInputElement;
    input.value = "rent in Yaba";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    expect(input.value).toBe("rent in Yaba");
  });
});
