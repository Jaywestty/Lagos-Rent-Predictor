import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { timeAgo } from "../lib/format";
import { duration, ScrapeProgress, type Run } from "./ScrapeProgress";

const BASE = import.meta.env.VITE_API_BASE_URL || "";
async function get(path: string) {
  const r = await fetch(`${BASE}${path}`, { credentials: "include" });
  if (!r.ok) throw Object.assign(new Error((await r.json().catch(() => ({}))).detail || "Failed"), { status: r.status });
  return r.json();
}

const STATUS_TONE: Record<string, string> = {
  success: "var(--good)", partial: "var(--warn)", interrupted: "var(--warn)", stopped: "var(--secondary)", failed: "var(--danger)", running: "var(--primary)", queued: "var(--secondary)",
};

function StatusBadge({ status }: { status: string }) {
  return <span className="badge"><span aria-hidden style={{ color: STATUS_TONE[status] ?? "var(--secondary)" }}>●</span>{status}</span>;
}

/** Active run plus last finished run. Falls back to the run list when the backend predates /scrape/active. */
async function fetchActive(): Promise<{ active: Run | null; last: Run | null }> {
  try {
    return await get("/api/admin/operations/scrape/active");
  } catch {
    const list = await get("/api/admin/operations/scraper-runs?page=1&page_size=10");
    const items: Run[] = list.items || [];
    return {
      active: items.find((r) => r.status === "running" || r.status === "queued") ?? null,
      last: items.find((r) => r.status !== "running" && r.status !== "queued") ?? null,
    };
  }
}

export function Operations() {
  const [confirming, setConfirming] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const qc = useQueryClient();
  const live = useQuery({
    queryKey: ["scrape-active"],
    queryFn: fetchActive,
    refetchInterval: (q) => (q.state.data?.active ? 3000 : 15000),
  });
  const active = live.data?.active ?? null;
  const last = live.data?.last ?? null;
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => get("/api/admin/operations/scraper-runs?page=1&page_size=10"), refetchInterval: active ? 10000 : false });
  const fresh = useQuery({ queryKey: ["fresh"], queryFn: () => get("/api/admin/operations/freshness?page=1&page_size=20") });
  const queue = useQuery({ queryKey: ["queue"], queryFn: () => get("/api/admin/operations/review-queue?page=1&page_size=10") });

  async function stopScrape(id: number) {
    const res = await fetch(`${BASE}/api/admin/operations/scrape/${id}/stop`, { method: "POST", credentials: "include", headers: { "X-CSRF-Token": localStorage.getItem("admin-csrf") || "" } });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Could not stop (${res.status})`);
    }
    await qc.invalidateQueries({ queryKey: ["scrape-active"] });
  }

  async function runScrape() {
    setStartError(null);
    const res = await fetch(`${BASE}/api/admin/operations/scrape`, { method: "POST", credentials: "include", headers: { "X-CSRF-Token": localStorage.getItem("admin-csrf") || "" } });
    setConfirming(false);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setStartError(body.detail || `Could not start (${res.status})`);
      return;
    }
    qc.invalidateQueries({ queryKey: ["scrape-active"] });
    qc.invalidateQueries({ queryKey: ["runs"] });
  }

  return (
    <div>
      <h1 className="text-[22px] font-semibold">Operations</h1>

      <div className="mt-4">
        {live.isPending && <div className="skeleton h-40" />}
        {active && <ScrapeProgress run={active} onStop={stopScrape} />}
        {!active && live.data && (
          <section className="card p-4 md:p-5" aria-label="Scraper">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-[17px] font-semibold">Scraper is idle</h2>
                {last ? (
                  <p className="muted mt-1 flex flex-wrap items-center gap-2 text-[13px]">
                    Last run #{last.id} <StatusBadge status={last.status} /> {last.completed_at ? `finished ${timeAgo(last.completed_at)}` : ""}
                    · {last.listings_added} new, {last.listings_updated} updated
                    {last.completed_at ? ` · took ${duration(new Date(last.completed_at).getTime() - new Date(last.started_at).getTime())}` : ""}
                  </p>
                ) : <p className="muted mt-1 text-[13px]">No runs yet.</p>}
              </div>
              {!confirming ? (
                <button type="button" className="btn" onClick={() => setConfirming(true)}>Run scrape now</button>
              ) : (
                <div role="alertdialog" aria-label="Confirm scrape" className="flex flex-wrap items-center gap-2">
                  <span className="text-[14px]">Start a scrape of the last 7 days now?</span>
                  <button type="button" className="btn" onClick={runScrape}>Start</button>
                  <button type="button" className="btn-ghost" onClick={() => setConfirming(false)}>Cancel</button>
                </div>
              )}
            </div>
            {startError && <p className="mt-2 text-[13px]" role="alert" style={{ color: "var(--danger)" }}>{startError}</p>}
          </section>
        )}
      </div>

      <section className="card mt-4 p-4">
        <h2 className="text-[17px] font-semibold">Run history</h2>
        {runs.isPending && <div className="skeleton mt-2 h-24" />}
        {runs.data && (
          <div className="overflow-x-auto">
            <table className="mt-2 w-full min-w-[560px] text-[14px]">
              <thead><tr className="muted text-left text-[12px]"><th className="py-2">Run</th><th>Started</th><th>By</th><th>Status</th><th className="text-right">New</th><th className="text-right">Updated</th><th className="text-right">Took</th></tr></thead>
              <tbody>
                {(runs.data.items || []).map((r: Run) => (
                  <tr key={r.id} className="border-t" style={{ borderColor: "var(--border)" }}>
                    <td className="py-2">#{r.id}</td>
                    <td>{timeAgo(r.started_at)}</td>
                    <td className="max-w-[160px] truncate" title={r.triggered_by}>{r.triggered_by}</td>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="tabular text-right">{r.listings_added}</td>
                    <td className="tabular text-right">{r.listings_updated}</td>
                    <td className="tabular text-right">{r.completed_at ? duration(new Date(r.completed_at).getTime() - new Date(r.started_at).getTime()) : "…"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card mt-4 p-4"><h2 className="text-[17px] font-semibold">Freshness by area</h2>
        {fresh.data && <table className="mt-2 w-full text-[14px]"><thead><tr className="muted text-left text-[12px]"><th className="py-2">Area</th><th>Listings</th><th>Freshness</th></tr></thead><tbody>{(fresh.data.items || []).map((r: { area: string; listing_count: number; freshness: string }) => <tr key={r.area} className="border-t" style={{ borderColor: "var(--border)" }}><td className="py-2">{r.area}</td><td className="tabular">{r.listing_count}</td><td>{r.freshness}</td></tr>)}</tbody></table>}
      </section>

      <section className="card mt-4 p-4"><h2 className="text-[17px] font-semibold">Review queue</h2>
        {queue.data && !(queue.data.items || []).length && <p className="muted mt-2 text-[14px]">Nothing to review.</p>}
        {queue.data && (queue.data.items || []).map((r: { id: number; raw_question: string; status: string }) => (
          <div key={r.id} className="mt-2 flex items-center justify-between border-t py-2 text-[14px]" style={{ borderColor: "var(--border)" }}><span>{r.raw_question} <span className="muted">({r.status})</span></span><button type="button" className="btn-ghost" onClick={async () => {
            await fetch(`${BASE}/api/admin/operations/review/${r.id}`, { method: "PATCH", credentials: "include", headers: { "Content-Type": "application/json", "X-CSRF-Token": localStorage.getItem("admin-csrf") || "" }, body: JSON.stringify({ review_state: "reviewed", review_note: "", flag_for_coverage: false }) });
            qc.invalidateQueries({ queryKey: ["queue"] });
          }}>Mark reviewed</button></div>
        ))}
      </section>
    </div>
  );
}
