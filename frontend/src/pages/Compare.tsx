import { useEffect, useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getAreas, getCompare } from "../services/api";
import { BUCKET_LABELS, nairaShort, timeAgo } from "../lib/format";
import type { BucketKey } from "../types/api";
import { PriceStrip, scaleFor } from "../components/charts/PriceStrip";
import { TrendChart } from "../components/charts/TrendChart";
import { ConfidenceBadge } from "../components/rent/RentAnswer";
import { confidenceFor } from "../lib/stats";
import { PageError, Skeleton } from "../components/states/States";

// First three validated categorical slots; each area keeps its colour by position in the picker.
const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const BEDS = [{ v: 0, l: "All sizes" }, { v: 1, l: "1 bed" }, { v: 2, l: "2 bed" }, { v: 3, l: "3 bed" }];
const BUCKETS: BucketKey[] = ["self_contain", "one_bedroom", "two_bedroom", "three_bedroom"];

export function Compare() {
  const [params, setParams] = useSearchParams();
  const picked = useMemo(() => (params.get("areas") || "").split(",").map((a) => a.trim()).filter(Boolean).slice(0, 3), [params]);
  const beds = Number(params.get("beds") || 0);
  const areas = useQuery({ queryKey: ["areas"], queryFn: getAreas, staleTime: 5 * 60_000 });
  const cmp = useQuery({ queryKey: ["compare", picked.join(","), beds], queryFn: () => getCompare(picked, beds || null), enabled: picked.length >= 2 });

  useEffect(() => { document.title = "Compare areas · Adugbo AI"; }, []);

  function setPicked(next: string[]) {
    const p = new URLSearchParams(params);
    if (next.length) p.set("areas", next.join(",")); else p.delete("areas");
    setParams(p, { replace: true });
  }
  function setBeds(v: number) {
    const p = new URLSearchParams(params);
    if (v) p.set("beds", String(v)); else p.delete("beds");
    setParams(p, { replace: true });
  }
  const options = (areas.data ?? []).map((a) => a.area).filter((a) => !picked.includes(a));
  const colorOf = (name: string) => COLORS[Math.max(0, picked.findIndex((p) => p.toLowerCase() === name.toLowerCase()))] ?? COLORS[0];
  const data = cmp.data;
  const scale = scaleFor(data?.areas.map((a) => a.focus) ?? []);
  const cheapest = data?.areas.filter((a) => a.focus?.median != null).sort((a, b) => (a.focus!.median! - b.focus!.median!))[0];

  return (
    <div className="mx-auto w-full max-w-[1080px] px-4 pb-12 pt-6 md:px-10 md:pt-10">
      <h1 className="text-[28px] font-bold">Compare areas</h1>
      <p className="muted mt-1 text-[15px]">Pick two or three areas to see rent side by side.</p>

      {/* Filters: one row above everything they scope */}
      <div className="mt-5 flex flex-wrap items-center gap-2">
        {picked.map((a, i) => (
          <span key={a} className="chip inline-flex items-center gap-2" style={{ borderColor: COLORS[i] }}>
            <span aria-hidden className="inline-block h-[10px] w-[10px] rounded-sm" style={{ background: COLORS[i] }} />{a}
            <button type="button" aria-label={`Remove ${a}`} onClick={() => setPicked(picked.filter((p) => p !== a))} className="muted">✕</button>
          </span>
        ))}
        {picked.length < 3 && (
          <select className="input" aria-label="Add an area" value="" onChange={(e) => e.target.value && setPicked([...picked, e.target.value])}>
            <option value="">+ Add area</option>
            {options.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        )}
        <span className="mx-1 hidden h-6 w-px sm:inline-block" style={{ background: "var(--border)" }} />
        {BEDS.map((b) => <button key={b.v} type="button" className="chip" aria-pressed={beds === b.v} onClick={() => setBeds(b.v)}>{b.l}</button>)}
      </div>

      <div className="mt-6">
        {picked.length < 2 && <p className="card p-6 text-[15px]">Add {2 - picked.length} more area{picked.length === 1 ? "" : "s"} to compare.</p>}
        {cmp.isPending && picked.length >= 2 && <Skeleton rows={3} />}
        {cmp.isError && <PageError message="The comparison did not load. Try again in a moment." />}
        {data && (
          <div className={cmp.isFetching ? "opacity-60 transition-opacity" : "transition-opacity"}>
            {cheapest && <p className="fade-in mb-4 text-[15px]">For <strong>{data.label.toLowerCase()}</strong>, <strong>{cheapest.area}</strong> has the lowest typical rent at <strong className="tabular">{nairaShort(cheapest.focus?.median)}</strong> a year.</p>}
            <div className="stagger grid gap-3 md:grid-cols-3">
              {data.areas.map((a) => (
                <article key={a.area} className="card p-4" style={{ borderTop: `3px solid ${colorOf(a.area)}` }}>
                  <div className="flex items-baseline justify-between gap-2">
                    <h2 className="text-[18px] font-semibold">{a.area}</h2>
                    <Link to={`/areas/${encodeURIComponent(a.area)}`} className="text-[12px] font-semibold hover:underline" style={{ color: "var(--primary)" }}>Guide →</Link>
                  </div>
                  <p className="tabular mt-2 text-[30px] font-bold leading-none">{nairaShort(a.focus?.median)}</p>
                  <p className="muted text-[12px]">typical rent a year · {a.focus?.count ?? 0} listings · newest {timeAgo(a.data_as_of)}</p>
                  <div className="mt-2"><ConfidenceBadge confidence={confidenceFor(a.focus?.count ?? 0)} /></div>
                  {a.trend.change_pct != null && <p className="mt-2 text-[13px] tabular">{a.trend.change_pct > 0 ? "▲" : "▼"} {Math.abs(a.trend.change_pct)}% since first week</p>}
                </article>
              ))}
            </div>

            <section className="card mt-4 p-4 md:p-6" aria-label="Price ranges">
              <h2 className="text-[17px] font-semibold">Price range · {data.label}</h2>
              {data.areas.map((a, i) => <PriceStrip key={a.area} label={a.area} stat={a.focus} scale={scale} color={colorOf(a.area)} delay={i * 80} />)}
            </section>

            <section className="card mt-4 p-4 md:p-6">
              <TrendChart title={`Typical rent week by week · ${data.label}`} series={data.areas.map((a) => ({ name: a.area, color: colorOf(a.area), trend: a.trend }))} />
            </section>

            <section className="card mt-4 overflow-x-auto p-4 md:p-6" aria-label="Comparison table">
              <h2 className="text-[17px] font-semibold">Typical rent by size</h2>
              <table className="mt-3 w-full min-w-[480px] text-[14px]">
                <thead><tr className="muted text-left text-[12px]"><th className="py-2">Size</th>{data.areas.map((a) => <th key={a.area} className="py-2 text-right">{a.area}</th>)}</tr></thead>
                <tbody>
                  {BUCKETS.map((b) => (
                    <tr key={b} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="py-2 font-semibold">{BUCKET_LABELS[b]}</td>
                      {data.areas.map((a) => <td key={a.area} className="tabular py-2 text-right">{a.breakdown[b] ? <>{nairaShort(a.breakdown[b]!.median)} <span className="muted text-[12px]">({a.breakdown[b]!.count})</span></> : <span className="muted">-</span>}</td>)}
                    </tr>
                  ))}
                  <tr className="border-t" style={{ borderColor: "var(--border)" }}>
                    <td className="py-2 font-semibold">All listings</td>
                    {data.areas.map((a) => <td key={a.area} className="tabular py-2 text-right font-semibold">{nairaShort(a.overall?.median)} <span className="muted text-[12px] font-normal">({a.listing_count})</span></td>)}
                  </tr>
                </tbody>
              </table>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
