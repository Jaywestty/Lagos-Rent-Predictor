import { useState } from "react";
import { nairaShort } from "../../lib/format";
import type { Summary } from "../../types/api";

type Props = {
  label: string;
  stat: Summary | null;
  scale: { min: number; max: number };
  color?: string;
  delay?: number;
};

/** A range bar (low to high, percentile when 10+ listings) with the median marked. Hover or focus shows the numbers. */
export function PriceStrip({ label, stat, scale, color = "var(--primary)", delay = 0 }: Props) {
  const [hover, setHover] = useState(false);
  if (!stat || stat.count === 0 || stat.min == null || stat.max == null) {
    return (
      <div className="flex min-h-[44px] items-center justify-between border-b py-2 text-[14px]" style={{ borderColor: "var(--border)" }}>
        <span className="font-semibold">{label}</span>
        <span className="muted text-[13px]">No listings yet</span>
      </div>
    );
  }
  const span = Math.max(scale.max - scale.min, 1);
  const left = ((stat.min - scale.min) / span) * 100;
  const width = Math.max(((stat.max - stat.min) / span) * 100, 1.5);
  const marker = stat.median != null ? ((stat.median - scale.min) / span) * 100 : null;
  const method = stat.range_method === "percentile" ? "most listings fall in this range" : "cheapest to dearest";
  return (
    <div className="border-b py-3" style={{ borderColor: "var(--border)" }}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[14px] font-semibold">{label}</span>
        <span className="tabular text-[14px] font-semibold">{nairaShort(stat.median)} <span className="muted text-[12px] font-normal">typical</span></span>
      </div>
      <div
        className="relative mt-2 h-6 rounded outline-none"
        tabIndex={0}
        role="img"
        aria-label={`${label}: ${nairaShort(stat.min)} to ${nairaShort(stat.max)} a year, typical ${nairaShort(stat.median)}, ${stat.count} listings`}
        onPointerEnter={() => setHover(true)}
        onPointerLeave={() => setHover(false)}
        onFocus={() => setHover(true)}
        onBlur={() => setHover(false)}
      >
        <div className="absolute inset-x-0 top-[9px] h-[6px] rounded" style={{ background: "var(--soft)" }} />
        <div className="grow-x absolute top-[9px] h-[6px] rounded" style={{ left: `${left}%`, width: `${Math.min(width, 100 - left)}%`, background: color, animationDelay: `${delay}ms`, opacity: hover ? 1 : 0.9 }} />
        {marker != null && <div className="fade-in absolute top-[3px] h-[18px] w-[3px] rounded" style={{ left: `calc(${marker}% - 1.5px)`, background: "var(--text)", animationDelay: `${delay + 300}ms`, boxShadow: "0 0 0 2px var(--surface)" }} />}
        {hover && (
          <div className="tooltip" style={{ left: `${Math.min(Math.max(marker ?? left, 10), 70)}%`, bottom: 28 }}>
            <strong className="tabular text-[13px]">{nairaShort(stat.median)}</strong> typical · {nairaShort(stat.min)} to {nairaShort(stat.max)} a year
            <div className="muted">{stat.count} listings · {method}</div>
          </div>
        )}
      </div>
      <div className="muted mt-1 flex justify-between text-[12px] tabular">
        <span>{nairaShort(stat.min)}</span>
        <span>{stat.count} listing{stat.count === 1 ? "" : "s"}</span>
        <span>{nairaShort(stat.max)}</span>
      </div>
    </div>
  );
}

export function scaleFor(stats: (Summary | null | undefined)[]): { min: number; max: number } {
  const mins = stats.map((s) => s?.min).filter((v): v is number => v != null);
  const maxs = stats.map((s) => s?.max).filter((v): v is number => v != null);
  if (!mins.length) return { min: 0, max: 1 };
  return { min: Math.min(...mins) * 0.9, max: Math.max(...maxs) * 1.05 };
}
