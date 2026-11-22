import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ScrapeProgress, type Run } from "../src/admin/ScrapeProgress";

const now = new Date().toISOString();
const base: Run = { id: 9, triggered_by: "admin:x", started_at: now, completed_at: null, status: "running", listings_added: 12, listings_updated: 3 };

describe("scrape progress", () => {
  it("shows step, page and counts from live progress", () => {
    render(<ScrapeProgress run={{ ...base, progress: {
      phase: "scraping", step: 3, total_steps: 6, site: "propertypro", category: "house", page_number: 2, max_pages: 30,
      page_done: 10, page_total: 25, started_at: now, heartbeat_at: now, saved: { added: 40, updated: 5, pages: 4 },
      counts: { already_have: 7, too_old: 11, fetch_failed: 0, pages_failed: 0 },
    } }} />);
    expect(screen.getByText("Step 3 of 6")).toBeTruthy();
    expect(screen.getByText("Listing 10 of 25 on page 2")).toBeTruthy();
    expect(screen.getByText("40")).toBeTruthy();
    expect(screen.getByRole("progressbar", { name: "Current page" }).getAttribute("aria-valuenow")).toBe("40");
  });

  it("falls back to saved counts for runs without progress and offers no stop button", () => {
    render(<ScrapeProgress run={base} onStop={vi.fn()} />);
    expect(screen.getByText("12")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Stop scrape" })).toBeNull();
  });

  it("asks before stopping and calls onStop", async () => {
    const onStop = vi.fn(async () => undefined);
    render(<ScrapeProgress onStop={onStop} run={{ ...base, progress: {
      phase: "scraping", step: 1, total_steps: 6, site: "npc", category: "houses", page_number: 1, max_pages: 30,
      page_done: 0, page_total: 20, started_at: now, heartbeat_at: now,
    } }} />);
    fireEvent.click(screen.getByRole("button", { name: "Stop scrape" }));
    fireEvent.click(screen.getByRole("button", { name: "Yes, stop" }));
    await vi.waitFor(() => expect(onStop).toHaveBeenCalledWith(9));
  });

  it("shows stopping state once a stop was requested", () => {
    render(<ScrapeProgress onStop={vi.fn()} run={{ ...base, stop_requested_at: now, progress: {
      phase: "scraping", step: 1, total_steps: 6, site: "npc", category: "houses", page_number: 1, max_pages: 30,
      page_done: 0, page_total: 20, started_at: now, heartbeat_at: now,
    } }} />);
    expect(screen.getByText(/Stopping after the listing it is on/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Stop scrape" })).toBeNull();
  });
});
