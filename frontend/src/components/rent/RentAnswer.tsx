import { useState } from "react";
import { Link } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import type { BucketKey, ChatAnswer, Summary } from "../../types/api";
import { BUCKET_LABELS, nairaShort, timeAgo } from "../../lib/format";
import { saveSearch, track } from "../../services/api";
import { PriceStrip, scaleFor } from "../charts/PriceStrip";

const CONFIDENCE = {
  normal: { label: "Solid data", tone: "var(--action)", bg: "var(--soft)", note: "10 or more listings" },
  limited: { label: "Limited data", tone: "var(--warn)", bg: "var(--sand)", note: "4 to 9 listings" },
  insufficient: { label: "Very little data", tone: "var(--danger)", bg: "var(--sand)", note: "3 or fewer listings" },
} as const;

export function ConfidenceBadge({ confidence, count }: { confidence: keyof typeof CONFIDENCE; count?: number }) {
  const c = CONFIDENCE[confidence] ?? CONFIDENCE.insufficient;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[12px] font-semibold" style={{ background: c.bg, color: c.tone }} title={c.note}>
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
        <circle cx="5" cy="5" r="4" fill="none" stroke="currentColor" strokeWidth="2" />
        {confidence !== "insufficient" && <path d={confidence === "normal" ? "M5 1a4 4 0 1 1 0 8a4 4 0 1 1 0-8z" : "M5 1a4 4 0 0 1 0 8z"} fill="currentColor" />}
      </svg>
      {c.label}{count != null ? ` · ${count} listing${count === 1 ? "" : "s"}` : ""}
    </span>
  );
}

// The backend uses percentiles per bucket once it has 10+ listings (PERCENTILE_MIN_COUNT).
function asSummary(s: { min: number | null; max: number | null; median: number | null; count: number } | null | undefined): Summary | null {
  return s ? { ...s, range_method: s.count >= 10 ? "percentile" : "min_max" } : null;
}

function Thumb({ up }: { up: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden style={up ? undefined : { transform: "scaleY(-1)" }}>
      <path d="M7 10v11H3V10zM7 10l5-7a2 2 0 0 1 3 2l-1 5h6a2 2 0 0 1 2 2l-2 8a2 2 0 0 1-2 1H7" />
    </svg>
  );
}

export function RentAnswer({ answer, question, onAsk }: { answer: ChatAnswer; question?: string; onAsk?: (q: string) => void }) {
  const qc = useQueryClient();
  const [saved, setSaved] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
  const keys: BucketKey[] = ["self_contain", "one_bedroom", "two_bedroom", "three_bedroom"];
  const rows = keys.map((k) => ({ key: k, stat: asSummary(answer.breakdown?.[k]) }));
  const shown = rows.filter((r) => r.stat && r.stat.count > 0);
  const scale = scaleFor([...rows.map((r) => r.stat), answer.headline ? { ...answer.headline, median: null, count: 1, range_method: "min_max" } : null]);
  const where = answer.area || answer.location;
  const freshness = answer.data_as_of ? timeAgo(answer.data_as_of) : null;
  const min = answer.headline?.min ?? null;
  const max = answer.headline?.max ?? null;
  const markerPct = answer.median != null && min != null && max != null && max > min ? ((answer.median - min) / (max - min)) * 100 : 50;

  async function onSave() {
    setSaved("saving");
    try {
      await saveSearch({ label: question || `Rent in ${where}`, query: question ?? null, area: where ?? null });
      setSaved("saved");
      qc.invalidateQueries({ queryKey: ["saved"] });
    } catch {
      setSaved("error");
    }
  }

  // Hide suggestions that repeat what was just asked (e.g. "what about a 2 bedroom?" after a 2 bedroom question).
  const asked = (question || "").toLowerCase();
  const beds = asked.match(/(\d)\s*-?\s*bed/)?.[1];
  const followUps = (answer.follow_ups || []).filter((q) => !(beds && q.toLowerCase().includes(`${beds} bedroom`))).slice(0, 2);

  function rate(value: "up" | "down") {
    setFeedback(value);
    track("answer_feedback", { useful: value === "up", area: where ?? null, request_id: answer.request_id || null });
  }

  return (
    <section className="card rise overflow-hidden" aria-label="Rent answer">
      <div className="space-y-5 p-5 md:p-7">
        <div className="flex flex-wrap items-center gap-2">
          <p className="kicker">{[where, answer.property_type, "per year"].filter(Boolean).join(" · ")}</p>
          <ConfidenceBadge confidence={answer.confidence} count={answer.listing_count} />
        </div>
        <div className="flex flex-wrap items-end gap-x-10 gap-y-3">
          <div>
            <p className="muted text-[13px]">Typical rent</p>
            <p className="figure mt-1 text-[48px] font-bold leading-none md:text-[60px]">{nairaShort(answer.median)}</p>
          </div>
          <div className="pb-1">
            <p className="muted text-[13px]">{answer.range_method === "percentile" ? "Most listings fall between" : "Listings range from"}</p>
            <p className="figure mt-1 text-[22px] font-semibold">{nairaShort(min)} – {nairaShort(max)}</p>
          </div>
        </div>
        {min != null && max != null && (
          <div role="img" aria-label={`Range ${nairaShort(min)} to ${nairaShort(max)}, typical ${nairaShort(answer.median)}`} className="relative h-6">
            <div className="absolute inset-x-0 top-[9px] h-[6px] rounded-full" style={{ background: "var(--line)" }} />
            <div className="grow-x absolute inset-x-0 top-[9px] h-[6px] rounded-full" style={{ background: "var(--action)" }} />
            <div className="fade-in absolute top-[2px] h-[20px] w-[3px] rounded" style={{ left: `calc(${markerPct}% - 1.5px)`, background: "var(--text)", boxShadow: "0 0 0 2px var(--surface)", animationDelay: "400ms" }} />
          </div>
        )}
        {answer.budget && (
          <div className="rounded-xl px-4 py-3 text-[14px] leading-relaxed" style={{ background: answer.budget.within_count ? "var(--soft)" : "var(--sand)" }}>
            <p>
              Your budget: <strong className="figure">{nairaShort(answer.budget.amount)}</strong> {answer.budget.period === "month" ? `a month (${nairaShort(answer.budget.yearly)} a year)` : "a year"}.{" "}
              {answer.budget.within_count > 0
                ? <><strong>{answer.budget.within_count} listing{answer.budget.within_count === 1 ? "" : "s"}</strong> at or under it{answer.budget.areas.length ? `, mostly in ${answer.budget.areas.slice(0, 3).map((a) => a.area).join(", ")}` : ""}.</>
                : <>No listings fit it yet{answer.budget.stretch_count ? `; ${answer.budget.stretch_count} are within 15% above it` : ""}.</>}
            </p>
          </div>
        )}
        <p className="muted text-[13px]">
          From {answer.listing_count} listing{answer.listing_count === 1 ? "" : "s"}{freshness ? `, newest ${freshness}` : ""}. Monthly prices are converted to yearly.
        </p>

        {shown.length > 1 && (
          <details className="group rounded-xl border px-4 py-1" style={{ borderColor: "var(--border)" }}>
            <summary className="flex min-h-[44px] cursor-pointer items-center justify-between text-[14px] font-semibold">
              Typical rent by size
              <svg className="transition-transform group-open:rotate-180" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M6 9l6 6 6-6" /></svg>
            </summary>
            <div className="pb-2">
              {rows.map((r, i) => <PriceStrip key={r.key} label={BUCKET_LABELS[r.key]} stat={r.stat} scale={scale} delay={i * 70} />)}
            </div>
          </details>
        )}

        <div className="flex flex-wrap gap-2">
          {where && <Link to={`/areas/${encodeURIComponent(where)}`} className="btn-ghost inline-flex items-center">{where} area guide</Link>}
          {where && <Link to={`/move-in?area=${encodeURIComponent(where)}${answer.median ? `&rent=${Math.round(answer.median)}` : ""}`} className="btn-ghost inline-flex items-center">Move-in cost</Link>}
          {where && <Link to={`/compare?areas=${encodeURIComponent(where)}`} className="btn-ghost inline-flex items-center">Compare</Link>}
          <button type="button" onClick={onSave} disabled={saved === "saving" || saved === "saved"} className="btn-ghost">
            {saved === "saved" ? "Saved ✓" : saved === "saving" ? "Saving..." : saved === "error" ? "Could not save, retry" : "Save search"}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t px-5 py-3 md:px-7" style={{ borderColor: "var(--line)" }}>
        {followUps.length > 0 && onAsk ? (
          <div className="flex flex-wrap gap-2" aria-label="Follow-up questions">
            {followUps.map((q) => <button key={q} type="button" className="chip" onClick={() => onAsk(q)}>{q}</button>)}
          </div>
        ) : <span />}
        <div className="flex items-center gap-2">
          <span className="muted text-[13px]">{feedback ? "Thanks for the feedback" : "Was this useful?"}</span>
          <button type="button" aria-label="Useful" aria-pressed={feedback === "up"} onClick={() => rate("up")} disabled={feedback !== null}
            className="grid h-9 w-9 place-items-center rounded-lg border" style={{ borderColor: "var(--border)", color: feedback === "up" ? "var(--action)" : "var(--secondary)" }}><Thumb up /></button>
          <button type="button" aria-label="Not useful" aria-pressed={feedback === "down"} onClick={() => rate("down")} disabled={feedback !== null}
            className="grid h-9 w-9 place-items-center rounded-lg border" style={{ borderColor: "var(--border)", color: feedback === "down" ? "var(--danger)" : "var(--secondary)" }}><Thumb up={false} /></button>
        </div>
      </div>
    </section>
  );
}
