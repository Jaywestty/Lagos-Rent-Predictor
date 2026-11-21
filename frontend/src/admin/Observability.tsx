import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { timeAgo } from "../lib/format";

const BASE = import.meta.env.VITE_API_BASE_URL || "";
async function get<T = any>(path: string): Promise<T> { // eslint-disable-line @typescript-eslint/no-explicit-any
  const r = await fetch(`${BASE}${path}`, { credentials: "include" });
  if (!r.ok) throw Object.assign(new Error((await r.json().catch(() => ({}))).detail || `Failed (${r.status})`), { status: r.status });
  return r.json();
}

const RANGES = [{ h: 24, l: "24 hours" }, { h: 168, l: "7 days" }, { h: 720, l: "30 days" }];
const LEVEL = { critical: { icon: "✖", c: "var(--danger)" }, warning: { icon: "▲", c: "var(--warn)" }, info: { icon: "ℹ", c: "var(--secondary)" } } as const;
const ms = (v: number | null | undefined) => (v == null ? "-" : v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`);
const n = (v: number | null | undefined) => (v == null ? "-" : Number(v).toLocaleString("en-NG"));
const bytes = (v: number) => (v > 1e9 ? `${(v / 1e9).toFixed(2)} GB` : `${(v / 1e6).toFixed(1)} MB`);

function Tile({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="card p-4">
      <p className="muted text-[12px] font-semibold">{label}</p>
      <p className="tabular text-[22px] font-bold" style={tone ? { color: tone } : undefined}>{value}</p>
      {sub && <p className="muted text-[12px]">{sub}</p>}
    </div>
  );
}

function Table({ cols, rows, empty = "Nothing recorded yet." }: { cols: { k: string; l: string; f?: (v: any, r: any) => React.ReactNode; right?: boolean }[]; rows: any[]; empty?: string }) { // eslint-disable-line @typescript-eslint/no-explicit-any
  if (!rows?.length) return <p className="muted text-[13px]">{empty}</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-[13px]">
        <thead><tr className="muted text-left text-[12px]">{cols.map((c) => <th key={c.k} className={`py-2 pr-3 ${c.right ? "text-right" : ""}`}>{c.l}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i} className="border-t" style={{ borderColor: "var(--border)" }}>{cols.map((c) => <td key={c.k} className={`py-2 pr-3 align-top ${c.right ? "tabular text-right" : ""}`}>{c.f ? c.f(r[c.k], r) : r[c.k] ?? "-"}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

/** Requests per bucket as bars, server errors overlaid in red. Hover a bar for its numbers. */
function TrafficChart({ points }: { points: { t: string; requests: number; errors: number; p95_ms: number | null; questions: number }[] }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!points.length) return <p className="muted text-[13px]">No traffic in this window yet.</p>;
  const max = Math.max(...points.map((p) => p.requests), 1);
  const p = hover != null ? points[hover] : null;
  return (
    <div className="relative">
      <div className="flex h-36 items-end gap-[2px]" role="img" aria-label="Requests over time">
        {points.map((pt, i) => (
          <div key={pt.t} className="relative flex h-full flex-1 items-end" onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
            <div className="w-full rounded-t transition-colors" style={{ height: `${(pt.requests / max) * 100}%`, minHeight: 2, background: hover === i ? "var(--forest)" : "var(--primary)" }} />
            {pt.errors > 0 && <div className="absolute bottom-0 w-full rounded-t" style={{ height: `${(pt.errors / max) * 100}%`, minHeight: 2, background: "var(--danger)" }} />}
          </div>
        ))}
      </div>
      <div className="muted mt-1 flex justify-between text-[11px]"><span>{new Date(points[0].t).toLocaleString("en-NG")}</span><span>{new Date(points.at(-1)!.t).toLocaleString("en-NG")}</span></div>
      {p && (
        <div className="tooltip" style={{ left: `${Math.min(Math.max(((hover! + 0.5) / points.length) * 100, 10), 70)}%`, top: 0 }}>
          <div className="muted">{new Date(p.t).toLocaleString("en-NG")}</div>
          <div><strong className="tabular">{p.requests}</strong> requests · <strong className="tabular">{p.questions}</strong> questions</div>
          <div><strong className="tabular" style={{ color: p.errors ? "var(--danger)" : undefined }}>{p.errors}</strong> server errors · p95 {ms(p.p95_ms)}</div>
        </div>
      )}
      <p className="muted mt-2 flex gap-4 text-[12px]"><span><span className="mr-1 inline-block h-[10px] w-[10px] rounded-sm align-middle" style={{ background: "var(--primary)" }} />Requests</span><span><span className="mr-1 inline-block h-[10px] w-[10px] rounded-sm align-middle" style={{ background: "var(--danger)" }} />Server errors</span></p>
    </div>
  );
}

function RequestLookup() {
  const [rid, setRid] = useState("");
  const [submitted, setSubmitted] = useState("");
  const q = useQuery({ queryKey: ["obs-req", submitted], queryFn: () => get(`/api/admin/observability/requests/${encodeURIComponent(submitted)}`), enabled: Boolean(submitted), retry: false });
  return (
    <div>
      <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); setSubmitted(rid.trim()); }}>
        <input className="input min-w-0 flex-1" placeholder="Paste a request id (X-Request-Id header, question log or error)" value={rid} onChange={(e) => setRid(e.target.value)} aria-label="Request id" />
        <button className="btn" type="submit">Trace</button>
      </form>
      {q.isFetching && <p className="pulse mt-2 text-[13px]">Looking up...</p>}
      {q.isError && <p className="mt-2 text-[13px]" role="alert">{(q.error as Error).message}</p>}
      {q.data && (
        <div className="fade-in mt-3 space-y-3 text-[13px]">
          {q.data.request?.[0] && <p><strong>{q.data.request[0].method} {q.data.request[0].route}</strong> → {q.data.request[0].status_code} in {ms(Number(q.data.request[0].duration_ms))}</p>}
          {q.data.question?.[0] && <p>Question: “{q.data.question[0].raw_question}” · {q.data.question[0].status} · {q.data.question[0].listing_count} listings</p>}
          <Table cols={[{ k: "stage", l: "Stage" }, { k: "duration_ms", l: "Time", right: true, f: (v) => ms(Number(v)) }, { k: "success", l: "OK", f: (v) => (v ? "✓" : "✖") }, { k: "error", l: "Error" }]} rows={q.data.stages} empty="No pipeline stages." />
          <Table cols={[{ k: "provider", l: "Provider" }, { k: "purpose", l: "Purpose" }, { k: "latency_ms", l: "Time", right: true, f: (v) => ms(Number(v)) }, { k: "prompt_tokens", l: "Tokens in", right: true }, { k: "completion_tokens", l: "Tokens out", right: true }, { k: "success", l: "OK", f: (v) => (v ? "✓" : "✖") }]} rows={q.data.ai_calls} empty="No AI or search calls." />
          {q.data.errors?.map((e: { error_type: string; message: string; stack: string }, i: number) => <details key={i} className="card p-3"><summary className="cursor-pointer font-semibold" style={{ color: "var(--danger)" }}>{e.error_type}: {e.message}</summary><pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-[11px]">{e.stack}</pre></details>)}
        </div>
      )}
    </div>
  );
}

export function Observability() {
  const [hours, setHours] = useState(24);
  const [errPage, setErrPage] = useState(1);
  const o = useQuery({ queryKey: ["obs", hours], queryFn: () => get(`/api/admin/observability/overview?hours=${hours}`), refetchInterval: 60_000 });
  const ts = useQuery({ queryKey: ["obs-ts", hours], queryFn: () => get(`/api/admin/observability/timeseries?hours=${hours}`), refetchInterval: 60_000 });
  const errors = useQuery({ queryKey: ["obs-err", errPage], queryFn: () => get(`/api/admin/observability/errors?page=${errPage}&page_size=10`) });
  const guides = useQuery({ queryKey: ["obs-guides"], queryFn: () => get("/api/admin/observability/guides") });
  const refresh = useMutation({
    mutationFn: async (area: string) => {
      const r = await fetch(`${BASE}/api/admin/observability/guides/${encodeURIComponent(area)}/refresh`, { method: "POST", credentials: "include", headers: { "X-CSRF-Token": localStorage.getItem("admin-csrf") || "" } });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Refresh failed");
      return r.json();
    },
    onSuccess: () => guides.refetch(),
  });

  const d = o.data;
  return (
    <div className={o.isFetching && d ? "opacity-80 transition-opacity" : "transition-opacity"}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[22px] font-semibold">Observability</h1>
        <div className="flex gap-2" role="group" aria-label="Time range">
          {RANGES.map((r) => <button key={r.h} type="button" className="chip" aria-pressed={hours === r.h} onClick={() => setHours(r.h)}>{r.l}</button>)}
        </div>
      </div>
      {o.isPending && <div className="mt-4 grid gap-3 sm:grid-cols-3">{Array.from({ length: 6 }, (_, i) => <div key={i} className="skeleton h-24" />)}</div>}
      {o.isError && <p className="card mt-4 p-4" role="alert">Could not load observability: {(o.error as Error).message}</p>}
      {d && (
        <div className="stagger mt-4 space-y-6">
          <section aria-label="Health">
            {d.health.length === 0 ? (
              <p className="card p-4 text-[14px]"><span style={{ color: "var(--good)" }}>●</span> All systems normal in the last {RANGES.find((r) => r.h === hours)?.l}.</p>
            ) : (
              <ul className="space-y-2">{d.health.map((h: { level: keyof typeof LEVEL; text: string }, i: number) => <li key={i} className="card flex items-center gap-2 p-3 text-[14px]"><span aria-hidden style={{ color: LEVEL[h.level].c }}>{LEVEL[h.level].icon}</span><strong className="capitalize">{h.level}</strong> {h.text}</li>)}</ul>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-[17px] font-semibold">Traffic and latency</h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Tile label="Requests" value={n(d.traffic.requests)} sub={`${n(d.traffic.unique_clients)} devices`} />
              <Tile label="Server error rate" value={`${d.traffic.error_rate}%`} sub={`${n(d.traffic.server_errors)} 5xx · ${n(d.traffic.client_errors)} 4xx`} tone={d.traffic.error_rate > 2 ? "var(--danger)" : undefined} />
              <Tile label="Latency p50 / p95" value={`${ms(d.traffic.p50_ms)} / ${ms(d.traffic.p95_ms)}`} sub="public API, all routes" />
              <Tile label="Questions" value={n(d.questions.total)} sub={`${n(d.questions.sessions)} sessions · p95 ${ms(d.questions.p95_ms)}`} />
            </div>
            <div className="card mt-3 p-4">{ts.data ? <TrafficChart points={ts.data.points} /> : <div className="skeleton h-36" />}</div>
            <div className="card mt-3 p-4"><h3 className="mb-2 text-[14px] font-semibold">Routes</h3>
              <Table cols={[{ k: "method", l: "" }, { k: "route", l: "Route" }, { k: "requests", l: "Requests", right: true }, { k: "errors", l: "5xx", right: true }, { k: "p50_ms", l: "p50", right: true, f: ms }, { k: "p95_ms", l: "p95", right: true, f: ms }]} rows={d.routes} />
            </div>
          </section>

          <section className="grid gap-3 lg:grid-cols-2">
            <div className="card p-4">
              <h2 className="mb-2 text-[17px] font-semibold">Answer quality</h2>
              <div className="grid grid-cols-3 gap-2 text-center text-[13px]">
                {[["Answered", d.questions.success, "var(--good)"], ["Low data", d.questions.low_data, "var(--warn)"], ["No data", d.questions.no_data, "var(--danger)"], ["Errors", d.questions.error, "var(--danger)"], ["Clarify / other", d.questions.other, undefined], ["Avg time", null, undefined]].map(([l, v, c]) => (
                  <div key={l as string} className="rounded-lg p-2" style={{ background: "var(--soft)" }}><p className="tabular text-[18px] font-bold" style={c ? { color: c as string } : undefined}>{l === "Avg time" ? ms(d.questions.avg_ms) : n(v as number)}</p><p className="muted">{l as string}</p></div>
                ))}
              </div>
              <h3 className="mb-1 mt-4 text-[14px] font-semibold">Areas people ask about with no or low data</h3>
              <Table cols={[{ k: "area", l: "Area" }, { k: "c", l: "Questions", right: true }]} rows={d.unanswered_areas} empty="None. Coverage matches demand." />
            </div>
            <div className="card p-4">
              <h2 className="mb-2 text-[17px] font-semibold">AI and search calls</h2>
              <Table cols={[{ k: "provider", l: "Provider" }, { k: "purpose", l: "Purpose" }, { k: "calls", l: "Calls", right: true }, { k: "failures", l: "Failed", right: true, f: (v) => <span style={{ color: v ? "var(--danger)" : undefined }}>{v}</span> }, { k: "p95_ms", l: "p95", right: true, f: ms }, { k: "prompt_tokens", l: "Tokens in", right: true, f: n }, { k: "completion_tokens", l: "Tokens out", right: true, f: n }]} rows={d.ai} />
              <h3 className="mb-1 mt-4 text-[14px] font-semibold">Pipeline stages</h3>
              <Table cols={[{ k: "stage", l: "Stage" }, { k: "runs", l: "Runs", right: true }, { k: "failures", l: "Failed", right: true }, { k: "avg_ms", l: "Avg", right: true, f: ms }, { k: "p95_ms", l: "p95", right: true, f: ms }]} rows={d.stages} />
            </div>
          </section>

          <section className="card p-4">
            <div className="mb-2 flex items-center justify-between"><h2 className="text-[17px] font-semibold">Errors</h2><span className="muted text-[12px]">{d.errors_by_source.map((e: { source: string; c: number }) => `${e.source} ${e.c}`).join(" · ") || "none in window"}</span></div>
            {errors.data?.items?.length ? (
              <div className="space-y-2">
                {errors.data.items.map((e: { id: number; source: string; route: string | null; error_type: string; message: string; stack: string | null; created_at: string; request_id: string | null }) => (
                  <details key={e.id} className="rounded-lg border p-3 text-[13px]" style={{ borderColor: "var(--border)" }}>
                    <summary className="cursor-pointer"><span className="badge mr-2">{e.source}</span><strong>{e.error_type}</strong>: {e.message?.slice(0, 140)} <span className="muted">· {timeAgo(e.created_at)}{e.route ? ` · ${e.route}` : ""}</span></summary>
                    {e.request_id && <p className="muted mt-2">Request id: <code>{e.request_id}</code></p>}
                    {e.stack && <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-[11px]">{e.stack}</pre>}
                  </details>
                ))}
                <div className="flex gap-2"><button type="button" className="btn-ghost" disabled={errPage <= 1} onClick={() => setErrPage((p) => p - 1)}>Prev</button><button type="button" className="btn-ghost" disabled={errPage * 10 >= (errors.data.total || 0)} onClick={() => setErrPage((p) => p + 1)}>Next</button><span className="muted self-center text-[12px]">{errors.data.total} total</span></div>
              </div>
            ) : <p className="muted text-[13px]">No errors recorded.</p>}
          </section>

          <section className="card p-4"><h2 className="mb-2 text-[17px] font-semibold">Trace a request</h2><RequestLookup /></section>

          <section>
            <h2 className="mb-2 text-[17px] font-semibold">Data and scraper</h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Tile label="Live listings" value={n(d.data.listings)} sub={`${n(d.data.areas)} areas · ${n(d.data.duplicates)} duplicates hidden`} />
              <Tile label="Added in last 7 days" value={n(d.data.added_7d)} sub={`newest listing ${timeAgo(d.data.newest_listing)}`} />
              <Tile label="Last scrape" value={d.scraper.last_run ? d.scraper.last_run.status : "never"} sub={d.scraper.last_run ? `${timeAgo(d.scraper.last_run.started_at)} · +${d.scraper.last_run.listings_added} new` : undefined} tone={d.scraper.last_run?.status === "failed" ? "var(--danger)" : undefined} />
              <Tile label="Next weekly run" value={new Date(d.scraper.next_scheduled_run).toLocaleDateString("en-NG", { weekday: "short", day: "numeric", month: "short" })} sub={`${d.scraper.runs} runs in 30 days · ${d.scraper.failed} failed`} />
            </div>
            <div className="mt-3 grid gap-3 lg:grid-cols-2">
              <div className="card p-4">
                <h3 className="mb-2 text-[14px] font-semibold">Data quality</h3>
                <Table cols={[{ k: "l", l: "Check" }, { k: "v", l: "Listings", right: true }]} rows={[
                  { l: "Rent with no price period (treated as yearly)", v: n(d.data.missing_period) },
                  { l: "Monthly price above ₦10M (likely yearly)", v: n(d.data.implausible_monthly) },
                  { l: "Yearly price below ₦300k (likely monthly)", v: n(d.data.implausible_yearly) },
                  { l: "No listing date", v: n(d.data.missing_listed_at) },
                ]} />
                <p className="muted mt-2 text-[12px]">By source: {d.data.by_source.map((s: { source: string; c: number }) => `${s.source} ${s.c}`).join(" · ")} · database {bytes(d.data.db_size_bytes)}</p>
              </div>
              <div className="card p-4">
                <h3 className="mb-2 text-[14px] font-semibold">Areas going stale (no new listing in 14 days)</h3>
                <Table cols={[{ k: "area", l: "Area" }, { k: "listing_count", l: "Listings", right: true }, { k: "last_ingested_at", l: "Last ingested", f: (v) => timeAgo(v) }]} rows={d.data.stale_areas} empty="None. Every area has fresh listings." />
              </div>
            </div>
          </section>

          <section className="grid gap-3 lg:grid-cols-2">
            <div className="card p-4">
              <h2 className="mb-2 text-[17px] font-semibold">Feature usage</h2>
              <div className="mb-3 grid grid-cols-2 gap-2 text-[13px]">
                <Tile label="Saved searches" value={n(d.engagement.saved_searches)} sub={`${n(d.engagement.saved_search_clients)} devices`} />
                <Tile label="Favourites" value={n(d.engagement.favorites)} sub={`${n(d.engagement.favorite_clients)} devices`} />
              </div>
              <Table cols={[{ k: "feature", l: "Feature" }, { k: "c", l: "Events", right: true }, { k: "clients", l: "Devices", right: true }]} rows={d.features} />
              <h3 className="mb-1 mt-3 text-[14px] font-semibold">Devices</h3>
              <p className="text-[13px]">{d.devices.map((x: { family: string; c: number }) => `${x.family} ${x.c}`).join(" · ") || "-"}</p>
            </div>
            <div className="card p-4">
              <h2 className="mb-2 text-[17px] font-semibold">Area guides and trends</h2>
              <p className="muted mb-2 text-[12px]">{n(d.engagement.guides)} guides cached · {n(d.engagement.snapshot_weeks)} weeks of price snapshots (latest {d.engagement.latest_snapshot_week ?? "none"})</p>
              <Table cols={[{ k: "area", l: "Area" }, { k: "source_count", l: "Sources", right: true }, { k: "generated_at", l: "Built", f: (v) => timeAgo(v) }, { k: "area", l: "", f: (v) => <button type="button" className="font-semibold" style={{ color: "var(--primary)" }} disabled={refresh.isPending} onClick={() => refresh.mutate(v)}>{refresh.isPending && refresh.variables === v ? "Rebuilding..." : "Rebuild"}</button> }]} rows={guides.data?.items ?? []} empty="No guides built yet. They build on first view, or run warm-guides." />
              {refresh.isError && <p className="mt-2 text-[13px]" role="alert">{(refresh.error as Error).message}</p>}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
