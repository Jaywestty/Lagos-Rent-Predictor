import { useState } from "react";
import { Link } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import type { BucketKey, ChatAnswer, Summary } from "../../types/api";
import { BUCKET_LABELS, nairaShort, timeAgo } from "../../lib/format";
import { saveSearch } from "../../services/api";
import { PriceStrip, scaleFor } from "../charts/PriceStrip";

const CONFIDENCE = {
  normal: { label: "Solid data", icon: "●", tone: "var(--good)", note: "10 or more listings" },
  limited: { label: "Limited data", icon: "◐", tone: "var(--warn)", note: "4 to 9 listings" },
  insufficient: { label: "Very little data", icon: "○", tone: "var(--danger)", note: "3 or fewer listings" },
} as const;

export function ConfidenceBadge({ confidence }: { confidence: keyof typeof CONFIDENCE }) {
  const c = CONFIDENCE[confidence] ?? CONFIDENCE.insufficient;
  return <span className="badge" title={c.note}><span aria-hidden style={{ color: c.tone }}>{c.icon}</span>{c.label}</span>;
}

// The backend uses percentiles per bucket once it has 10+ listings (PERCENTILE_MIN_COUNT).
function asSummary(s: { min: number | null; max: number | null; median: number | null; count: number } | null | undefined): Summary | null {
  return s ? { ...s, range_method: s.count >= 10 ? "percentile" : "min_max" } : null;
}

export function RentAnswer({ answer, question, onAsk }: { answer: ChatAnswer; question?: string; onAsk?: (q: string) => void }) {
  const qc = useQueryClient();
  const [saved, setSaved] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const keys: BucketKey[] = ["self_contain", "one_bedroom", "two_bedroom", "three_bedroom"];
  const rows = keys.map((k) => ({ key: k, stat: asSummary(answer.breakdown?.[k]) }));
  const scale = scaleFor([...rows.map((r) => r.stat), answer.headline ? { ...answer.headline, median: null, count: 1, range_method: "min_max" } : null]);
  const shown = rows.filter((r) => r.stat && r.stat.count > 0);
  const where = answer.area || answer.location;
  const freshness = answer.data_as_of ? timeAgo(answer.data_as_of) : answer.last_updated_days != null ? `${answer.last_updated_days} days ago` : null;

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

  return (
    <section className="card rise overflow-hidden" aria-label="Rent answer">
      <div className="p-4 md:p-6" style={{ background: "linear-gradient(180deg, var(--soft), transparent)" }}>
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-[12px] font-semibold uppercase tracking-wide muted">{where ? `${where} · ` : ""}{answer.property_type || "All property types"} · per year</p>
          <ConfidenceBadge confidence={answer.confidence} />
        </div>
        <div className="mt-2 flex flex-wrap items-end gap-x-6 gap-y-2">
          <div>
            <p className="muted text-[13px]">Typical rent (median)</p>
            <p className="tabular text-[40px] font-bold leading-none md:text-[48px]">{nairaShort(answer.median)}</p>
          </div>
          <div className="pb-1">
            <p className="muted text-[13px]">{answer.range_method === "percentile" ? "Most listings fall between" : "Listings range from"}</p>
            <p className="tabular text-[20px] font-semibold">{nairaShort(answer.headline?.min)} – {nairaShort(answer.headline?.max)}</p>
          </div>
        </div>
        <p className="muted mt-3 text-[13px]">
          From <strong style={{ color: "var(--text)" }}>{answer.listing_count} listing{answer.listing_count === 1 ? "" : "s"}</strong>
          {freshness ? <>, newest {freshness}</> : null}. Monthly prices are converted to yearly.
        </p>
      </div>

      {shown.length > 0 && (
        <div className="px-4 pb-2 md:px-6">
          {rows.map((r, i) => <PriceStrip key={r.key} label={BUCKET_LABELS[r.key]} stat={r.stat} scale={scale} delay={i * 70} />)}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 border-t px-4 py-3 md:px-6" style={{ borderColor: "var(--border)" }}>
        {where && <Link to={`/areas/${encodeURIComponent(where)}`} className="btn-ghost inline-flex items-center">Area guide</Link>}
        {where && <Link to={`/move-in?area=${encodeURIComponent(where)}${answer.median ? `&rent=${Math.round(answer.median)}` : ""}`} className="btn-ghost inline-flex items-center">Move-in cost</Link>}
        {where && <Link to={`/compare?areas=${encodeURIComponent(where)}`} className="btn-ghost inline-flex items-center">Compare</Link>}
        <button type="button" onClick={onSave} disabled={saved === "saving" || saved === "saved"} className="btn-ghost">
          {saved === "saved" ? "Saved ✓" : saved === "saving" ? "Saving..." : saved === "error" ? "Could not save, retry" : "Save search"}
        </button>
      </div>

      {answer.follow_ups?.length > 0 && onAsk && (
        <div className="flex flex-wrap gap-2 px-4 pb-4 md:px-6" aria-label="Follow-up questions">
          {answer.follow_ups.slice(0, 3).map((q) => <button key={q} type="button" className="chip" onClick={() => onAsk(q)}>{q}</button>)}
        </div>
      )}
    </section>
  );
}
