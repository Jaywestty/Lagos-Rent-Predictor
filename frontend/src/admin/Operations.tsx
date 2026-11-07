import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

const BASE = import.meta.env.VITE_API_BASE_URL || "";
async function get(path: string) {
  const r = await fetch(`${BASE}${path}`, { credentials: "include" });
  if (!r.ok) throw new Error("Failed");
  return r.json();
}

export function Operations() {
  const [confirming, setConfirming] = useState(false);
  const [jobId, setJobId] = useState<number | null>(null);
  const qc = useQueryClient();
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => get("/admin/operations/scraper-runs?page=1&page_size=10") });
  const fresh = useQuery({ queryKey: ["fresh"], queryFn: () => get("/admin/operations/freshness?page=1&page_size=20") });
  const queue = useQuery({ queryKey: ["queue"], queryFn: () => get("/admin/operations/review-queue?page=1&page_size=10") });

  useEffect(() => {
    if (jobId == null) return;
    const t = setInterval(() => { qc.invalidateQueries({ queryKey: ["runs"] }); }, 4000);
    return () => clearInterval(t);
  }, [jobId, qc]);

  async function runScrape() {
    const res = await fetch(`${BASE}/admin/operations/scrape`, { method: "POST", credentials: "include", headers: { "X-CSRF-Token": localStorage.getItem("admin-csrf") || "" } });
    if (!res.ok) { setConfirming(false); return; }
    const body = await res.json();
    setJobId(body.job_id);
    setConfirming(false);
    qc.invalidateQueries({ queryKey: ["runs"] });
  }

  return (
    <div>
      <h1 className="text-[22px] font-semibold">Operations</h1>
      <section className="card mt-4 p-4" aria-label="Scraper">
        <h2 className="text-[17px] font-semibold">Scraper</h2>
        <p className="text-[15px]">Status: {jobId ? `job ${jobId} running or queued` : "idle"}</p>
        {!confirming ? <button type="button" onClick={() => setConfirming(true)} className="mt-2 min-h-[48px] rounded px-4 text-[14px] font-semibold text-white" style={{ background: "var(--primary)" }}>Run scrape now</button>
          : <div role="alertdialog" aria-label="Confirm scrape" className="mt-2 border p-3"><p>Start a scrape now? This will use live sites.</p><div className="mt-2 flex gap-2"><button type="button" onClick={runScrape} className="min-h-[48px] rounded px-4 text-[14px] font-semibold text-white" style={{ background: "var(--primary)" }}>Confirm</button><button type="button" onClick={() => setConfirming(false)} className="min-h-[48px] rounded border px-4 text-[14px] font-semibold">Cancel</button></div></div>}
      </section>
      <section className="card mt-4 p-4"><h2 className="text-[17px] font-semibold">Run history</h2>
        {runs.isPending && <p className="pulse">Loading...</p>}
        {runs.data && <table className="mt-2 w-full text-[15px]"><thead><tr><th className="text-left">Started</th><th className="text-left">By</th><th className="text-left">Status</th></tr></thead><tbody>{(runs.data.items || []).map((r: { id: number; started_at: string; triggered_by: string; status: string }) => <tr key={r.id} className="border-t"><td>{String(r.started_at).slice(0, 19)}</td><td>{r.triggered_by}</td><td>{r.status}</td></tr>)}</tbody></table>}
      </section>
      <section className="card mt-4 p-4"><h2 className="text-[17px] font-semibold">Freshness by area</h2>
        {fresh.data && <table className="mt-2 w-full text-[15px]"><thead><tr><th className="text-left">Area</th><th className="text-left">Listings</th><th className="text-left">Freshness</th></tr></thead><tbody>{(fresh.data.items || []).map((r: { area: string; listing_count: number; freshness: string }) => <tr key={r.area} className="border-t"><td>{r.area}</td><td>{r.listing_count}</td><td>{r.freshness}</td></tr>)}</tbody></table>}
      </section>
      <section className="card mt-4 p-4"><h2 className="text-[17px] font-semibold">Review queue</h2>
        {queue.data && (queue.data.items || []).map((r: { id: number; raw_question: string; status: string }) => (
          <div key={r.id} className="mt-2 flex items-center justify-between border-t py-2 text-[15px]"><span>{r.raw_question} <span style={{ color: "var(--secondary)" }}>({r.status})</span></span><button type="button" className="min-h-[44px] rounded border px-3 text-[14px] font-semibold" onClick={async () => {
            await fetch(`${BASE}/admin/operations/review/${r.id}`, { method: "PATCH", credentials: "include", headers: { "Content-Type": "application/json", "X-CSRF-Token": localStorage.getItem("admin-csrf") || "" }, body: JSON.stringify({ review_state: "reviewed", review_note: "", flag_for_coverage: false }) });
            qc.invalidateQueries({ queryKey: ["queue"] });
          }}>Mark reviewed</button></div>
        ))}
      </section>
    </div>
  );
}
