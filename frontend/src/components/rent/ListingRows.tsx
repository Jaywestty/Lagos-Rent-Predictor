import { useMemo, useState } from "react";
import { nairaShort, SOURCE_LABELS, timeAgo } from "../../lib/format";
import { track } from "../../services/api";
import type { Listing } from "../../types/api";
import { FavoriteButton } from "./ListingCard";

/** How a listing compares with the typical rent of the answer it belongs to. Within 5% counts as typical. */
export function priceContext(price: number | null | undefined, typical: number | null | undefined) {
  if (price == null || typical == null || typical <= 0) return null;
  const diff = price - typical;
  if (Math.abs(diff) / typical <= 0.05) return { label: "At typical", tone: "var(--secondary)", bg: "var(--line)" };
  const amount = nairaShort(Math.abs(diff));
  return diff < 0
    ? { label: `${amount} below typical`, tone: "var(--action)", bg: "var(--soft)" }
    : { label: `${amount} above typical`, tone: "var(--clay)", bg: "var(--sand)" };
}

type Sort = "cheapest" | "newest";

/** The listings behind an answer as a compact ledger: price, title, details, and how each compares. */
export function ListingRows({ listings, typical, title }: { listings: Listing[]; typical: number | null | undefined; title?: string }) {
  const [sort, setSort] = useState<Sort>("cheapest");
  const sorted = useMemo(() => {
    const xs = [...listings];
    if (sort === "cheapest") xs.sort((a, b) => (a.annual_price_ngn ?? Infinity) - (b.annual_price_ngn ?? Infinity));
    else xs.sort((a, b) => (b.listed_at ?? "").localeCompare(a.listed_at ?? ""));
    return xs;
  }, [listings, sort]);
  if (!listings.length) return null;

  return (
    <section aria-label="Matching listings" className="fade-in space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-display text-[18px] font-bold">{title ?? `The ${listings.length} listing${listings.length === 1 ? "" : "s"} behind this answer`}</h3>
        <div role="group" aria-label="Sort listings" className="flex gap-0.5 rounded-xl p-1" style={{ background: "var(--line)" }}>
          {(["cheapest", "newest"] as Sort[]).map((s) => (
            <button key={s} type="button" aria-pressed={sort === s} onClick={() => setSort(s)}
              className="min-h-[34px] rounded-lg px-3 text-[13px] font-semibold capitalize transition-colors"
              style={sort === s ? { background: "var(--surface)", color: "var(--text)", boxShadow: "0 1px 2px rgba(0,0,0,.06)" } : { color: "var(--secondary)" }}>{s}</button>
          ))}
        </div>
      </div>
      <ul className="card stagger overflow-hidden">
        {sorted.map((l) => {
          const ctx = priceContext(l.annual_price_ngn, typical);
          const monthly = /\/month$/.test(l.price);
          const suspicious = monthly && l.annual_price_ngn != null && l.annual_price_ngn / 12 > 10_000_000;
          return (
            <li key={l.url} className="flex items-center gap-2 border-b last:border-b-0" style={{ borderColor: "var(--line)" }}>
              <a href={l.url} target="_blank" rel="noopener noreferrer" onClick={() => track("listing_click", { source: l.source ?? null })}
                className="grid min-w-0 flex-1 grid-cols-[76px_minmax(0,1fr)] items-center gap-x-4 gap-y-1 py-3.5 pl-4 transition-colors hover:bg-[color:var(--bg)] sm:grid-cols-[96px_minmax(0,1fr)_auto] md:pl-5">
                <span className="figure row-span-2 text-[19px] font-bold sm:row-span-1 sm:text-[20px]">{l.annual_price_ngn != null ? nairaShort(l.annual_price_ngn) : l.price}</span>
                <span className="min-w-0">
                  <span className="block truncate text-[14px] font-semibold">{l.title || "Untitled listing"}</span>
                  <span className="muted block truncate text-[12px]">
                    {[l.subarea || l.area, l.bedrooms != null ? `${l.bedrooms} bed` : null, l.bathrooms != null ? `${l.bathrooms} bath` : null, `listed ${timeAgo(l.listed_at)}`, l.source ? SOURCE_LABELS[l.source] ?? l.source : null].filter(Boolean).join(" · ")}
                  </span>
                  {suspicious && <span className="mt-0.5 block text-[12px] font-semibold" style={{ color: "var(--warn)" }}>Listed as {l.price.replace("NGN ", "₦")}; the period may be wrong</span>}
                </span>
                {ctx && (
                  <span className="justify-self-start whitespace-nowrap rounded-full px-2.5 py-1 text-[12px] font-bold sm:justify-self-end" style={{ background: ctx.bg, color: ctx.tone }}>{ctx.label}</span>
                )}
              </a>
              <span className="pr-3 md:pr-4"><FavoriteButton listing={l} /></span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
