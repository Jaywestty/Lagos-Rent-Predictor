import { useQuery } from "@tanstack/react-query";

const BASE = import.meta.env.VITE_API_BASE_URL || "";
async function get(path: string) {
  const r = await fetch(`${BASE}${path}`, { credentials: "include" });
  if (!r.ok) throw new Error("Failed");
  return r.json();
}

function LineChart({ data }: { data: { day: string; c: number }[] }) {
  if (!data.length) return <p className="text-[15px]">No data</p>;
  const max = Math.max(...data.map((d) => d.c), 1);
  const points = data.map((d, i) => `${(i / Math.max(data.length - 1, 1)) * 100},${90 - (d.c / max) * 80}`).join(" ");
  return <svg viewBox="0 0 100 100" className="h-32 w-full" role="img" aria-label="Questions per day"><polyline points={points} fill="none" stroke="var(--primary)" strokeWidth="2" /></svg>;
}
function BarChart({ data }: { data: { area: string; c: number }[] }) {
  if (!data.length) return <p className="text-[15px]">No data</p>;
  const max = Math.max(...data.map((d) => d.c), 1);
  return <div className="space-y-2">{data.slice(0, 10).map((d) => <div key={d.area} className="flex items-center gap-2 text-[13px]"><span className="w-28 truncate">{d.area}</span><div className="h-3 flex-1 rounded" style={{ background: "var(--soft)" }}><div className="h-3 rounded" style={{ width: `${(d.c / max) * 100}%`, background: "var(--primary)" }} /></div><span className="tabular">{d.c}</span></div>)}</div>;
}

export function Analytics() {
  const overview = useQuery({ queryKey: ["overview"], queryFn: () => get("/admin/analytics/overview") });
  const qot = useQuery({ queryKey: ["qot"], queryFn: () => get("/admin/analytics/questions-over-time?page=1&page_size=30") });
  const areas = useQuery({ queryKey: ["areas"], queryFn: () => get("/admin/analytics/top-areas?page=1&page_size=10") });
  const o = overview.data || {};
  return (
    <div>
      <h1 className="text-[22px] font-semibold">Analytics</h1>
      {overview.isPending && <p className="pulse mt-4">Loading KPIs...</p>}
      {overview.isError && <p role="alert">Could not load analytics.</p>}
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-label="Key metrics">
        <div className="card p-4"><p className="text-[12px] font-semibold">Total questions</p><p className="tabular text-[20px] font-semibold">{o.total_questions ?? "-"}</p></div>
        <div className="card p-4"><p className="text-[12px] font-semibold">Sessions</p><p className="tabular text-[20px] font-semibold">{o.sessions ?? "-"}</p></div>
        <div className="card p-4"><p className="text-[12px] font-semibold">Top area</p><p className="text-[20px] font-semibold">{o.top_area?.area || "-"} <span className="tabular">({o.top_area?.count ?? 0})</span></p></div>
        <div className="card p-4"><p className="text-[12px] font-semibold">No or low data</p><p className="tabular text-[20px] font-semibold">{o.no_or_low_data_percentage ?? 0}%</p><p className="text-[13px]">no data {o.no_data_count ?? 0} · low data {o.low_data_count ?? 0}</p></div>
        <div className="card p-4"><p className="text-[12px] font-semibold">Average latency (completed)</p><p className="tabular text-[20px] font-semibold">{Math.round(o.average_latency_ms ?? 0)} ms</p></div>
        <div className="card p-4"><p className="text-[12px] font-semibold">Valid listings</p><p className="tabular text-[20px] font-semibold">{o.valid_listing_count ?? "-"}</p><p className="text-[13px]">last scrape {o.last_scrape_at || "never"}</p></div>
      </div>
      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <section className="card p-4"><h2 className="text-[17px] font-semibold">Questions per day</h2>{qot.data ? <LineChart data={(qot.data.items || []).slice().reverse()} /> : <p className="pulse">Loading...</p>}</section>
        <section className="card p-4"><h2 className="text-[17px] font-semibold">Top areas</h2>{areas.data ? <BarChart data={areas.data.items || []} /> : <p className="pulse">Loading...</p>}</section>
      </div>
      <section className="card mt-6 p-4">
        <h2 className="text-[17px] font-semibold">Recent questions</h2>
        <RecentTable />
      </section>
    </div>
  );
}

function RecentTable() {
  const params = new URLSearchParams(location.search);
  const page = Math.max(1, Number(params.get("page") || "1"));
  const status = params.get("status") || "";
  const q = useQuery({ queryKey: ["recent", page, status], queryFn: () => get(`/admin/analytics/recent-questions?page=${page}&page_size=10${status ? `&status=${encodeURIComponent(status)}` : ""}`) });
  if (q.isPending) return <p className="pulse mt-2">Loading table...</p>;
  if (q.isError) return <p role="alert">Could not load recent questions.</p>;
  const items = q.data.items || [];
  if (!items.length) return <p>No questions yet.</p>;
  return (
    <div>
      <div className="mt-2 flex gap-2">
        <label className="text-[13px] font-semibold">Status <input aria-label="Status filter" defaultValue={status} onBlur={(e) => { location.search = new URLSearchParams({ page: "1", status: e.target.value }).toString(); }} className="ml-1 rounded border px-2 py-1 text-[13px]" /></label>
        <span className="text-[13px]">Total {q.data.total}</span>
      </div>
      <table className="mt-3 w-full text-[15px]"><thead><tr><th className="text-left">When</th><th className="text-left">Question</th><th className="text-left">Area</th><th className="text-left">Status</th></tr></thead>
        <tbody>{items.map((r: { id: number; created_at: string; raw_question: string; parsed_location: string | null; status: string }) => <tr key={r.id} className="border-t"><td>{String(r.created_at).slice(0, 19)}</td><td>{r.raw_question}</td><td>{r.parsed_location || "-"}</td><td>{r.status}</td></tr>)}</tbody></table>
      <div className="mt-3 flex gap-2">
        <button type="button" disabled={page <= 1} onClick={() => { location.search = new URLSearchParams({ page: String(page - 1), status }).toString(); }} className="min-h-[44px] rounded border px-3 text-[14px] font-semibold">Prev</button>
        <button type="button" onClick={() => { location.search = new URLSearchParams({ page: String(page + 1), status }).toString(); }} className="min-h-[44px] rounded border px-3 text-[14px] font-semibold">Next</button>
      </div>
    </div>
  );
}
