import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { addFavorite, deleteFavorite, getFavorites, track } from "../../services/api";
import { nairaShort, SOURCE_LABELS, timeAgo } from "../../lib/format";
import type { Listing } from "../../types/api";

export function useFavorites() {
  return useQuery({ queryKey: ["favorites"], queryFn: getFavorites, staleTime: 60_000, retry: false });
}

export function FavoriteButton({ listing }: { listing: Listing }) {
  const qc = useQueryClient();
  const favs = useFavorites();
  const existing = favs.data?.find((f) => f.url === listing.url);
  const [popKey, setPopKey] = useState(0);
  const m = useMutation({
    mutationFn: async (): Promise<unknown> => (existing ? deleteFavorite(existing.id) : addFavorite(listing)),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["favorites"] }),
  });
  const on = Boolean(existing);
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={on ? "Remove from favourites" : "Save to favourites"}
      disabled={m.isPending}
      onClick={() => { setPopKey((k) => k + 1); m.mutate(); }}
      className="grid h-10 w-10 shrink-0 place-items-center rounded-full border"
      style={{ borderColor: on ? "var(--danger)" : "var(--border)", color: on ? "var(--danger)" : "var(--secondary)" }}
    >
      <svg key={popKey} className={popKey ? "pop" : ""} width="18" height="18" viewBox="0 0 24 24" fill={on ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" aria-hidden>
        <path d="M12 21s-7.5-4.6-9.6-9.2C.9 8.3 3.1 4.5 6.9 4.5c2.1 0 3.6 1.2 5.1 3 1.5-1.8 3-3 5.1-3 3.8 0 6 3.8 4.5 7.3C19.5 16.4 12 21 12 21z" />
      </svg>
    </button>
  );
}

export function ListingCard({ listing }: { listing: Listing }) {
  const source = listing.source ? SOURCE_LABELS[listing.source] || listing.source : null;
  const annual = listing.annual_price_ngn;
  const monthly = /\/month$/.test(listing.price);
  return (
    <article className="card lift flex flex-col gap-2 p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="tabular text-[20px] font-bold leading-tight">{annual != null ? nairaShort(annual) : listing.price}<span className="muted text-[13px] font-normal"> /year</span></p>
          {monthly && <p className="muted text-[12px]">Listed as {listing.price.replace("NGN ", "₦")}</p>}
          {/* Same rule as the loader's quality report: monthly over ₦10M is usually a yearly price tagged wrong. */}
          {monthly && annual != null && annual / 12 > 10_000_000 && <p className="mt-1 text-[12px] font-semibold" style={{ color: "var(--warn)" }}>▲ Price period looks wrong, check on site</p>}
        </div>
        <FavoriteButton listing={listing} />
      </div>
      <h3 className="line-clamp-2 text-[14px] font-semibold leading-snug">{listing.title || "Untitled listing"}</h3>
      <p className="muted text-[13px]">{[listing.subarea, listing.area].filter(Boolean).join(", ")}</p>
      <div className="flex flex-wrap gap-2 text-[12px]">
        {listing.bedrooms != null && <span className="badge">{listing.bedrooms} bed</span>}
        {listing.bathrooms != null && <span className="badge">{listing.bathrooms} bath</span>}
        {listing.property_type && <span className="badge">{listing.property_type}</span>}
      </div>
      <div className="mt-auto flex items-center justify-between gap-2 pt-2 text-[12px]">
        <span className="muted">{source ? `${source} · ` : ""}listed {timeAgo(listing.listed_at)}</span>
        <a href={listing.url} target="_blank" rel="noopener noreferrer" onClick={() => track("listing_click", { source: listing.source ?? null })} className="font-semibold underline-offset-2 hover:underline" style={{ color: "var(--primary)" }}>
          View on site ↗
        </a>
      </div>
    </article>
  );
}

export function ListingGrid({ listings, empty = "No listings to show." }: { listings: Listing[]; empty?: string }) {
  if (!listings.length) return <p className="muted text-[14px]">{empty}</p>;
  return <div className="stagger grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{listings.map((l) => <ListingCard key={l.url} listing={l} />)}</div>;
}
