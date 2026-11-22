import { useEffect, useState } from "react";

export type RunProgress = {
  phase: "starting" | "scraping" | "finishing" | "done" | "failed" | "stopped";
  step: number;
  total_steps: number;
  site: string | null;
  category: string | null;
  page_number: number;
  max_pages: number | null;
  page_done: number;
  page_total: number;
  started_at: string;
  heartbeat_at?: string;
  saved?: { added: number; updated: number; pages: number };
  counts?: Record<string, number>;
};

export type Run = {
  id: number;
  triggered_by: string;
  started_at: string;
  completed_at: string | null;
  status: string;
  listings_added: number;
  listings_updated: number;
  listings_too_old?: number;
  error_count?: number;
  log_excerpt?: string | null;
  progress?: RunProgress | null;
  stop_requested_at?: string | null;
};

const SITE_NAMES: Record<string, string> = { npc: "NigeriaPropertyCentre", propertypro: "PropertyPro" };
const PHASE_TEXT: Record<string, string> = {
  starting: "Getting ready",
  scraping: "Collecting listings",
  finishing: "Final checks: removing duplicates and updating price trends",
  done: "Finished",
  failed: "Stopped with an error",
  stopped: "Stopped",
};
const STALL_SECONDS = 180;

function useNow(intervalMs = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(t);
  }, [intervalMs]);
  return now;
}

export function duration(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function Bar({ value, label }: { value: number | null; label: string }) {
  const pct = value == null ? null : Math.max(0, Math.min(100, value * 100));
  return (
    <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct == null ? undefined : Math.round(pct)}
      className="relative h-3 w-full overflow-hidden rounded-full" style={{ background: "var(--soft)" }}>
      {pct == null ? (
        <div className="indeterminate absolute inset-y-0 w-1/3 rounded-full" style={{ background: "var(--primary)" }} />
      ) : (
        <div className="h-3 rounded-full transition-[width] duration-700 ease-out" style={{ width: `${pct}%`, background: "var(--primary)" }} />
      )}
    </div>
  );
}

function Count({ label, value, tone }: { label: string; value: number | undefined; tone?: string }) {
  return (
    <div className="rounded-lg p-3" style={{ background: "var(--soft)" }}>
      <p className="tabular text-[20px] font-bold" style={tone ? { color: tone } : undefined}>{(value ?? 0).toLocaleString("en-NG")}</p>
      <p className="muted text-[12px]">{label}</p>
    </div>
  );
}

/** Live view of a running scrape. Works with detailed progress, and falls back to saved counts for older runs. */
export function ScrapeProgress({ run, onStop }: { run: Run; onStop?: (id: number) => Promise<void> }) {
  const now = useNow();
  const [confirmStop, setConfirmStop] = useState(false);
  const [stopError, setStopError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const stopping = Boolean(run.stop_requested_at);

  async function stop() {
    if (!onStop) return;
    setSending(true);
    setStopError(null);
    try {
      await onStop(run.id);
      setConfirmStop(false);
    } catch (e) {
      setStopError((e as Error).message);
    } finally {
      setSending(false);
    }
  }
  const p = run.progress ?? null;
  const elapsed = now - new Date(p?.started_at || run.started_at).getTime();
  const beat = p?.heartbeat_at ? (now - new Date(p.heartbeat_at).getTime()) / 1000 : null;
  const stalled = beat != null && beat > STALL_SECONDS;
  const added = p?.saved?.added ?? run.listings_added ?? 0;
  const updated = p?.saved?.updated ?? run.listings_updated ?? 0;

  const stepFraction = p && p.total_steps ? (Math.max(p.step - 1, 0) + (p.page_total ? p.page_done / p.page_total : 0) * 0.5) / p.total_steps : null;

  return (
    <section className="card rise p-4 md:p-5" aria-label="Scrape in progress" aria-live="polite">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="relative flex h-3 w-3" aria-hidden>
            {!stalled && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: "var(--primary)" }} />}
            <span className="relative inline-flex h-3 w-3 rounded-full" style={{ background: stalled ? "var(--warn)" : "var(--primary)" }} />
          </span>
          <h2 className="text-[17px] font-semibold">Scraping now · run #{run.id}</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <p className="muted text-[13px] tabular">Running for {duration(elapsed)}{beat != null ? ` · last update ${duration(beat * 1000)} ago` : ""}</p>
          {onStop && !stopping && !confirmStop && (
            <button type="button" className="btn-ghost" style={{ borderColor: "var(--danger)", color: "var(--danger)" }} onClick={() => setConfirmStop(true)}>Stop scrape</button>
          )}
        </div>
      </div>

      {confirmStop && !stopping && (
        <div role="alertdialog" aria-label="Confirm stop" className="fade-in mt-3 flex flex-wrap items-center gap-2 rounded-lg p-3 text-[14px]" style={{ background: "var(--soft)" }}>
          <span>Stop this scrape? Everything saved so far is kept, and the next run carries on from here.</span>
          <button type="button" className="btn" style={{ background: "var(--danger)" }} disabled={sending} onClick={stop}>{sending ? "Stopping..." : "Yes, stop"}</button>
          <button type="button" className="btn-ghost" onClick={() => setConfirmStop(false)}>Keep running</button>
        </div>
      )}
      {stopping && (
        <p className="fade-in mt-3 rounded-lg p-3 text-[13px] font-semibold" role="status" style={{ background: "var(--soft)" }}>
          Stopping after the listing it is on. Saving what was collected and tidying up; this takes under a minute.
        </p>
      )}
      {stopError && <p className="mt-2 text-[13px]" role="alert" style={{ color: "var(--danger)" }}>{stopError}</p>}

      {stalled && (
        <p className="mt-3 rounded-lg p-3 text-[13px]" role="alert" style={{ background: "var(--soft)", color: "var(--warn)" }}>
          No update for {duration(beat! * 1000)}. The scraper may have stopped (laptop asleep or the server restarted). Pages already saved are kept.
        </p>
      )}

      {p ? (
        <div className="mt-4 space-y-4">
          <p className="text-[14px] font-semibold">{PHASE_TEXT[p.phase] ?? p.phase}</p>
          <div>
            <div className="mb-1 flex justify-between text-[13px]">
              <span>{p.site ? `${SITE_NAMES[p.site] ?? p.site} · ${p.category}` : "Preparing"}</span>
              <span className="muted tabular">Step {Math.max(p.step, 0)} of {p.total_steps || "?"}</span>
            </div>
            <div className="flex gap-1" aria-hidden>
              {Array.from({ length: p.total_steps || 1 }, (_, i) => (
                <div key={i} className={`h-2 flex-1 rounded-full ${i === p.step - 1 && p.phase === "scraping" ? "pulse" : ""}`}
                  style={{ background: i < p.step - 1 || p.phase === "finishing" || p.phase === "done" ? "var(--primary)" : i === p.step - 1 ? "var(--forest)" : "var(--soft)" }} />
              ))}
            </div>
            <div className="mt-2"><Bar value={p.phase === "finishing" ? null : stepFraction} label="Overall progress" /></div>
            <p className="muted mt-1 text-[12px]">Each step stops on its own once listings are older than the 7-day window, so steps can be short or long.</p>
          </div>
          {p.phase === "scraping" && p.page_total > 0 && (
            <div>
              <div className="mb-1 flex justify-between text-[13px]">
                <span>Listing {p.page_done} of {p.page_total} on page {p.page_number}</span>
                <span className="muted">up to {p.max_pages} pages</span>
              </div>
              <Bar value={p.page_done / p.page_total} label="Current page" />
            </div>
          )}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            <Count label="New listings saved" value={added} tone="var(--good)" />
            <Count label="Updated" value={updated} />
            <Count label="Skipped, already saved" value={p.counts?.already_have} />
            <Count label="Older than 7 days" value={p.counts?.too_old} />
            <Count label="Failed to load" value={(p.counts?.fetch_failed ?? 0) + (p.counts?.pages_failed ?? 0)} tone={(p.counts?.fetch_failed ?? 0) + (p.counts?.pages_failed ?? 0) ? "var(--warn)" : undefined} />
          </div>
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          <Bar value={null} label="Scrape running" />
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Count label="New listings saved" value={added} tone="var(--good)" />
            <Count label="Updated" value={updated} />
          </div>
          <p className="muted text-[12px]">{run.log_excerpt || "Waiting for the first page to finish."} This run does not report step-by-step progress, so only saved counts are shown.</p>
        </div>
      )}
    </section>
  );
}
