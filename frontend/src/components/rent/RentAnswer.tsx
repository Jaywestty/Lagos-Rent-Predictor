import type { ChatAnswer } from "../../types/api";
import { formatNaira, formatNairaRange } from "../../lib/format";
import { stripPosition } from "../../lib/stats";

function Row({ label, stat, scale }: { label: string; stat: { min: number | null; max: number | null; median: number | null; count: number } | null; scale: { min: number; max: number } }) {
  if (!stat || stat.count === 0) return <div className="flex min-h-[44px] items-center justify-between border-b py-2 text-[15px]"><span>{label}</span><span className="text-[13px]" style={{ color: "var(--secondary)" }}>No data available</span></div>;
  const pos = stat.min != null && stat.max != null && stat.median != null ? stripPosition(stat.min, stat.median, stat.max, scale.min, scale.max) : { left: 0, width: 100, marker: 50 };
  return (
    <div className="border-b py-3">
      <div className="flex items-center justify-between gap-3 text-[15px]">
        <span className="font-semibold">{label}</span>
        <span className="tabular text-[13px]" style={{ color: "var(--secondary)" }}>{formatNairaRange(stat.min, stat.max)} per year</span>
      </div>
      <div className="relative mt-2 h-2 rounded" role="img" aria-label={`${label} range strip`} style={{ background: "var(--soft)" }}>
        <div className="absolute top-0 h-2 rounded" style={{ left: `${pos.left}%`, width: `${pos.width}%`, background: "var(--primary)" }} />
        <div className="absolute top-[-4px] h-4 w-1 rounded" style={{ left: `${pos.marker}%`, background: "var(--text)" }} />
      </div>
      <div className="mt-1 flex justify-between text-[12px] font-semibold tabular" style={{ color: "var(--secondary)" }}>
        <span>{formatNaira(stat.min)}</span><span>Median {formatNaira(stat.median)}</span><span>{formatNaira(stat.max)} · {stat.count} listings</span>
      </div>
    </div>
  );
}

export function RentAnswer({ answer }: { answer: ChatAnswer }) {
  const all = [answer.breakdown?.one_bedroom?.min, answer.breakdown?.two_bedroom?.min, answer.breakdown?.three_bedroom?.min, answer.headline?.min].filter((v): v is number => v != null);
  const allMax = [answer.breakdown?.one_bedroom?.max, answer.breakdown?.two_bedroom?.max, answer.breakdown?.three_bedroom?.max, answer.headline?.max].filter((v): v is number => v != null);
  const scale = { min: Math.min(...all, 0), max: Math.max(...allMax, 1) };
  const confIcon = answer.confidence === "normal" ? "●" : answer.confidence === "limited" ? "◐" : "○";
  const confPattern = answer.confidence === "normal" ? "solid disc" : answer.confidence === "limited" ? "half disc" : "hollow disc";
  return (
    <section className="card rise p-4 md:p-6" aria-label="Rent answer">
      <p className="text-[12px] font-semibold uppercase tracking-wide" style={{ color: "var(--secondary)" }}>Annual rent · NGN</p>
      <h2 className="tabular text-[26px] font-bold leading-tight md:text-[28px]">{formatNairaRange(answer.headline?.min ?? null, answer.headline?.max ?? null)} per year</h2>
      <p className="mt-1 flex flex-wrap items-center gap-2 text-[13px]" style={{ color: "var(--secondary)" }}>
        <span aria-hidden>{confIcon}</span><span>{answer.confidence} confidence · {confPattern}</span>
        {answer.median != null && <span className="tabular font-semibold" style={{ color: "var(--text)" }}>Median {formatNaira(answer.median)} per year</span>}
      </p>
      <div className="mt-3">
        <Row label="Self contain" stat={answer.breakdown?.self_contain ?? null} scale={scale} />
        <Row label="1 bedroom" stat={answer.breakdown?.one_bedroom ?? null} scale={scale} />
        <Row label="2 bedroom" stat={answer.breakdown?.two_bedroom ?? null} scale={scale} />
        <Row label="3 bedroom" stat={answer.breakdown?.three_bedroom ?? null} scale={scale} />
      </div>
      <p className="mt-3 text-[13px]" style={{ color: "var(--secondary)" }}>
        Based on {answer.listing_count} listings{answer.last_updated_days != null ? `, updated ${answer.last_updated_days} days ago` : ""}. Range method: {answer.range_method === "percentile" ? "10th to 90th percentile" : "minimum to maximum"}.
      </p>
      {answer.follow_ups?.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2" aria-label="Follow ups">
          {answer.follow_ups.slice(0, 3).map((q) => <span key={q} className="rounded-full border px-3 py-1 text-[13px]" style={{ borderColor: "var(--border)" }}>{q}</span>)}
        </div>
      )}
    </section>
  );
}
