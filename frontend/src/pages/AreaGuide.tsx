import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getArea, getAreaListings, getGuide, getTrend, saveSearch } from "../services/api";
import { BUCKET_LABELS, nairaShort, timeAgo } from "../lib/format";
import type { BucketKey } from "../types/api";
import { PriceStrip, scaleFor } from "../components/charts/PriceStrip";
import { TrendChart } from "../components/charts/TrendChart";
import { ListingGrid } from "../components/rent/ListingCard";
import { ConfidenceBadge } from "../components/rent/RentAnswer";
import { MoveInCalculator } from "../components/rent/MoveInCalculator";
import { PageError, Skeleton } from "../components/states/States";

const BUCKETS: BucketKey[] = ["self_contain", "one_bedroom", "two_bedroom", "three_bedroom"];
const BEDS_FOR: Record<string, number | null> = { all: null, self_contain: null, one_bedroom: 1, two_bedroom: 2, three_bedroom: 3 };
const ICONS: Record<string, string> = { overview: "◎", commute: "⇄", flooding: "☂", power: "⚡", security: "◈", amenities: "⌂", lifestyle: "☺" };

export function AreaGuide() {
  const { area = "" } = useParams();
  const qc = useQueryClient();
  const [bucket, setBucket] = useState<string>("all");
  const [saved, setSaved] = useState(false);
  const stats = useQuery({ queryKey: ["area", area], queryFn: () => getArea(area), retry: false });
  const guide = useQuery({ queryKey: ["guide", area], queryFn: () => getGuide(area), staleTime: 10 * 60_000, retry: false });
  const trend = useQuery({ queryKey: ["trend", area, bucket], queryFn: () => getTrend(area, bucket), enabled: stats.isSuccess });
  const listings = useQuery({ queryKey: ["area-listings", area, bucket], queryFn: () => getAreaListings(area, BEDS_FOR[bucket]), enabled: stats.isSuccess });

  useEffect(() => { document.title = `${area} rent and area guide · Lagos Rent`; }, [area]);

  if (stats.isPending) return <Shell><Skeleton rows={4} /></Shell>;
  if (stats.isError) return <Shell><PageError message={`We have no recent listings for ${area} yet. Try another area from the list.`} /><Link to="/areas" className="btn mt-4 inline-flex items-center">All areas</Link></Shell>;

  const s = stats.data;
  const scale = scaleFor(BUCKETS.map((b) => s.breakdown[b]));
  const g = guide.data?.guide;
  const sources = guide.data?.sources ?? [];

  return (
    <Shell>
      <nav className="muted text-[13px]"><Link to="/areas" className="hover:underline">Area guides</Link> / {s.area}</nav>
      <header className="stagger mt-2 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[32px] font-bold leading-tight">{s.area}</h1>
          <p className="muted mt-1 flex flex-wrap items-center gap-2 text-[14px]"><ConfidenceBadge confidence={s.confidence} /> {s.listing_count} listings · newest {timeAgo(s.data_as_of)}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link to={`/compare?areas=${encodeURIComponent(s.area)}`} className="btn-ghost inline-flex items-center">Compare</Link>
          <button type="button" className="btn" disabled={saved} onClick={async () => { await saveSearch({ label: `Rent in ${s.area}`, area: s.area }); setSaved(true); qc.invalidateQueries({ queryKey: ["saved"] }); }}>{saved ? "Saved ✓" : "Save area"}</button>
        </div>
      </header>

      <section className="mt-6 grid gap-4 lg:grid-cols-[1.1fr_1fr]">
        <div className="card rise p-4 md:p-6">
          <p className="muted text-[13px]">Typical rent per year, all sizes</p>
          <p className="tabular text-[40px] font-bold leading-none">{nairaShort(s.overall?.median)}</p>
          <p className="muted mt-1 text-[13px] tabular">{s.overall?.range_method === "percentile" ? "Most listings" : "Listings"} between {nairaShort(s.overall?.min)} and {nairaShort(s.overall?.max)}</p>
          <div className="mt-3">{BUCKETS.map((b, i) => <PriceStrip key={b} label={BUCKET_LABELS[b]} stat={s.breakdown[b]} scale={scale} delay={i * 70} />)}</div>
        </div>
        <div className="card rise p-4 md:p-6">
          <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label="Trend for">
            {["all", ...BUCKETS].map((b) => <button key={b} type="button" className="chip" aria-pressed={bucket === b} onClick={() => setBucket(b)}>{b === "all" ? "All" : BUCKET_LABELS[b]}</button>)}
          </div>
          {trend.data ? (
            <>
              <TrendChart title={`Typical rent week by week · ${trend.data.label}`} series={[{ name: s.area, color: "var(--primary)", trend: trend.data }]} />
              {trend.data.change_pct != null && (
                <p className="mt-2 text-[14px]"><strong className="tabular" style={{ color: trend.data.change_pct > 0 ? "var(--danger)" : "var(--good)" }}>{trend.data.change_pct > 0 ? "▲" : "▼"} {Math.abs(trend.data.change_pct)}%</strong> <span className="muted">since the first week on record</span></p>
              )}
            </>
          ) : <div className="skeleton h-48" />}
        </div>
      </section>

      <section className="mt-8" aria-label="Living here">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-[22px] font-semibold">Living in {s.area}</h2>
          {guide.data?.generated_at && <p className="muted text-[12px]">From {sources.length} web sources · updated {timeAgo(guide.data.generated_at)}{guide.data.stale ? " (refresh pending)" : ""}</p>}
        </div>
        {guide.isPending && <div className="mt-3"><p className="muted mb-2 text-[13px]">Reading local sources about {s.area}... this takes a few seconds the first time.</p><Skeleton rows={3} /></div>}
        {(guide.isError || guide.data?.guide_error) && <p className="card mt-3 p-4 text-[14px]">The area notes are not available right now. Rent figures above are unaffected.</p>}
        {g && (
          <>
            <div className="stagger mt-3 grid gap-3 md:grid-cols-2">
              {g.sections.filter((sec) => sec.text).map((sec) => (
                <article key={sec.key} className={`card p-4 ${sec.key === "overview" ? "md:col-span-2" : ""}`}>
                  <h3 className="flex items-center gap-2 text-[15px] font-semibold"><span aria-hidden className="grid h-7 w-7 place-items-center rounded-lg text-[14px]" style={{ background: "var(--soft)", color: "var(--primary)" }}>{ICONS[sec.key] ?? "•"}</span>{sec.title}</h3>
                  <p className="mt-2 text-[14px] leading-relaxed">{sec.text}</p>
                  {sec.sources.length > 0 && <p className="mt-2 flex flex-wrap gap-1 text-[12px]">{sec.sources.map((n) => { const src = sources.find((x) => x.n === n); return src ? <a key={n} href={src.url} target="_blank" rel="noopener noreferrer" className="badge hover:underline" title={src.title}>[{n}]</a> : null; })}</p>}
                </article>
              ))}
            </div>
            {g.sections.some((sec) => !sec.text) && <p className="muted mt-2 text-[12px]">Not covered by sources yet: {g.sections.filter((sec) => !sec.text).map((sec) => sec.title.toLowerCase()).join(", ")}.</p>}
            {g.watch_outs.length > 0 && (
              <div className="card mt-3 p-4" style={{ borderColor: "var(--warn)" }}>
                <h3 className="text-[15px] font-semibold">Watch out for</h3>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-[14px]">{g.watch_outs.map((w) => <li key={w}>{w}</li>)}</ul>
              </div>
            )}
            {sources.length > 0 && (
              <details className="mt-3 text-[13px]">
                <summary className="cursor-pointer font-semibold">Sources ({sources.length})</summary>
                <ol className="mt-2 list-decimal space-y-1 pl-5">{sources.map((src) => <li key={src.n}><a href={src.url} target="_blank" rel="noopener noreferrer" className="hover:underline">{src.title || src.url}</a></li>)}</ol>
              </details>
            )}
          </>
        )}
      </section>

      <section className="mt-8">
        <h2 className="mb-3 text-[22px] font-semibold">Newest listings{bucket !== "all" ? ` · ${BUCKET_LABELS[bucket]}` : ""}</h2>
        {listings.data ? <ListingGrid listings={listings.data} empty="No recent listings of this size." /> : <Skeleton rows={2} />}
      </section>

      <section className="mt-8">
        <h2 className="mb-3 text-[22px] font-semibold">What it costs to move in</h2>
        <MoveInCalculator initialRent={(bucket !== "all" ? s.breakdown[bucket as BucketKey]?.median : null) ?? s.overall?.median} area={s.area} />
      </section>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto w-full max-w-[1080px] px-4 pb-12 pt-6 md:px-10 md:pt-8">{children}</div>;
}
