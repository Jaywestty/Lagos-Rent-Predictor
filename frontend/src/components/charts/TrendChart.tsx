import { useMemo, useState } from "react";
import { nairaShort } from "../../lib/format";
import type { Trend } from "../../types/api";

type Series = { name: string; color: string; trend: Trend };

const W = 640;
const H = 220;
const PAD = { l: 52, r: 16, t: 16, b: 28 };

function fmtWeek(iso: string) {
  return new Date(iso + "T00:00:00").toLocaleDateString("en-NG", { day: "numeric", month: "short" });
}

/**
 * Weekly median rent. One series needs no legend (the heading names it); several get a legend
 * plus a direct label at each line end. A crosshair snaps to the nearest week and lists every series.
 */
export function TrendChart({ series, title }: { series: Series[]; title: string }) {
  const [idx, setIdx] = useState<number | null>(null);
  const weeks = useMemo(() => Array.from(new Set(series.flatMap((s) => s.trend.points.map((p) => p.week_start)))).sort(), [series]);
  const values = series.flatMap((s) => s.trend.points.map((p) => p.median_ngn)).filter((v): v is number => v != null);

  if (weeks.length < 2 || !values.length) {
    const only = series[0]?.trend.points.at(-1);
    return (
      <div className="rounded-xl border border-dashed p-4 text-[14px]" style={{ borderColor: "var(--border)" }}>
        <p className="font-semibold">{title}</p>
        <p className="muted mt-1">
          The trend line starts once there are two weeks of data. The weekly scrape adds a point every Monday.
          {only?.median_ngn != null && <> This week&apos;s typical rent: <strong className="tabular" style={{ color: "var(--text)" }}>{nairaShort(only.median_ngn)}</strong> from {only.listing_count} listings.</>}
        </p>
      </div>
    );
  }

  const lo = Math.min(...values) * 0.92;
  const hi = Math.max(...values) * 1.06;
  const x = (i: number) => PAD.l + (i / (weeks.length - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo || 1)) * (H - PAD.t - PAD.b);
  const ticks = [lo, (lo + hi) / 2, hi];

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (weeks.length - 1));
    setIdx(Math.max(0, Math.min(weeks.length - 1, i)));
  }

  return (
    <figure className="relative">
      <figcaption className="mb-2 flex flex-wrap items-center gap-3 text-[14px]">
        <span className="font-semibold">{title}</span>
        {series.length > 1 && series.map((s) => (
          <span key={s.name} className="muted inline-flex items-center gap-1 text-[12px]"><span className="inline-block h-[2px] w-4 rounded" style={{ background: s.color }} />{s.name}</span>
        ))}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${title}, typical rent each week`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={PAD.l - 8} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--secondary)">{nairaShort(t)}</text>
          </g>
        ))}
        {[0, weeks.length - 1].map((i) => (
          <text key={i} x={x(i)} y={H - 8} textAnchor={i === 0 ? "start" : "end"} fontSize={11} fill="var(--secondary)">{fmtWeek(weeks[i])}</text>
        ))}
        {series.map((s) => {
          const pts = weeks.map((w, i) => ({ i, p: s.trend.points.find((p) => p.week_start === w) })).filter((d) => d.p?.median_ngn != null);
          const d = pts.map((pt, k) => `${k ? "L" : "M"}${x(pt.i)},${y(pt.p!.median_ngn!)}`).join(" ");
          const last = pts.at(-1);
          return (
            <g key={s.name}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" className="draw" style={{ ["--len" as string]: 2000 }} />
              {pts.map((pt) => <circle key={pt.i} cx={x(pt.i)} cy={y(pt.p!.median_ngn!)} r={idx === pt.i ? 5 : 3} fill={s.color} stroke="var(--surface)" strokeWidth={2} />)}
              {series.length > 1 && last && <text x={x(last.i) - 6} y={y(last.p!.median_ngn!) - 8} textAnchor="end" fontSize={11} fontWeight={600} fill="var(--text)">{s.name}</text>}
            </g>
          );
        })}
        {idx != null && <line x1={x(idx)} x2={x(idx)} y1={PAD.t} y2={H - PAD.b} stroke="var(--secondary)" strokeWidth={1} strokeDasharray="3 3" />}
        <rect x={PAD.l} y={0} width={W - PAD.l - PAD.r} height={H} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setIdx(null)} />
      </svg>
      {idx != null && (
        <div className="tooltip" style={{ left: `${Math.min(Math.max((x(idx) / W) * 100, 12), 72)}%`, top: 24 }}>
          <div className="muted">Week of {fmtWeek(weeks[idx])}</div>
          {series.map((s) => {
            const p = s.trend.points.find((pt) => pt.week_start === weeks[idx]);
            return (
              <div key={s.name} className="flex items-center gap-2">
                <span className="inline-block h-[2px] w-3 rounded" style={{ background: s.color }} />
                <strong className="tabular">{nairaShort(p?.median_ngn)}</strong>
                <span className="muted">{series.length > 1 ? s.name : "typical rent"} · {p?.listing_count ?? 0} listings</span>
              </div>
            );
          })}
        </div>
      )}
    </figure>
  );
}
