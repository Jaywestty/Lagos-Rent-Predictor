import { useEffect } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteFavorite, deleteSaved, getSaved } from "../services/api";
import { nairaShort, SOURCE_LABELS, timeAgo } from "../lib/format";
import { useFavorites } from "../components/rent/ListingCard";
import { PageError, Skeleton } from "../components/states/States";

export function Saved() {
  const qc = useQueryClient();
  const saved = useQuery({ queryKey: ["saved"], queryFn: getSaved });
  const favs = useFavorites();
  const delSaved = useMutation({ mutationFn: deleteSaved, onSuccess: () => qc.invalidateQueries({ queryKey: ["saved"] }) });
  const delFav = useMutation({ mutationFn: deleteFavorite, onSuccess: () => qc.invalidateQueries({ queryKey: ["favorites"] }) });

  useEffect(() => { document.title = "Saved · Adugbo AI"; }, []);

  return (
    <div className="mx-auto w-full max-w-[1080px] px-4 pb-12 pt-6 md:px-10 md:pt-10">
      <h1 className="text-[28px] font-bold">Saved</h1>
      <p className="muted mt-1 text-[14px]">Kept on this device only. No account needed. Clearing your browser data removes them.</p>

      <section className="mt-6">
        <h2 className="text-[20px] font-semibold">Saved searches</h2>
        <div className="mt-3">
          {saved.isPending && <Skeleton rows={2} />}
          {saved.isError && <PageError message="Saved searches did not load." />}
          {saved.data && !saved.data.length && <p className="muted text-[14px]">Nothing saved yet. Use “Save search” on an answer or “Save area” on a guide.</p>}
          <div className="stagger grid gap-3 sm:grid-cols-2">
            {saved.data?.map((s) => (
              <article key={s.id} className="card lift flex items-start justify-between gap-3 p-4">
                <div className="min-w-0">
                  <h3 className="text-[15px] font-semibold">{s.label}</h3>
                  {s.current ? (
                    <p className="mt-1 text-[14px]"><strong className="tabular">{nairaShort(s.current.median_ngn)}</strong> <span className="muted">typical rent now · {s.current.listing_count} listings · newest {timeAgo(s.current.data_as_of)}</span></p>
                  ) : <p className="muted mt-1 text-[13px]">Saved {timeAgo(s.created_at)}</p>}
                  <div className="mt-2 flex flex-wrap gap-3 text-[13px] font-semibold" style={{ color: "var(--primary)" }}>
                    {s.query && <Link to={`/?q=${encodeURIComponent(s.query)}`} className="hover:underline">Ask again</Link>}
                    {s.area && <Link to={`/areas/${encodeURIComponent(s.area)}`} className="hover:underline">Area guide</Link>}
                  </div>
                </div>
                <button type="button" className="btn-ghost" aria-label={`Delete ${s.label}`} onClick={() => delSaved.mutate(s.id)}>Remove</button>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-8">
        <h2 className="text-[20px] font-semibold">Favourite listings</h2>
        <div className="mt-3">
          {favs.isPending && <Skeleton rows={2} />}
          {favs.data && !favs.data.length && <p className="muted text-[14px]">Tap the heart on any listing to keep it here.</p>}
          <div className="stagger grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {favs.data?.map((f) => (
              <article key={f.id} className="card lift flex flex-col gap-2 p-4">
                <div className="flex items-start justify-between gap-2">
                  <p className="tabular text-[20px] font-bold">{f.annual_price_ngn != null ? nairaShort(f.annual_price_ngn) : f.price_text}<span className="muted text-[13px] font-normal"> /year</span></p>
                  {!f.still_listed && <span className="badge" style={{ color: "var(--warn)" }}>No longer in our data</span>}
                </div>
                <h3 className="line-clamp-2 text-[14px] font-semibold">{f.title || "Listing"}</h3>
                <p className="muted text-[13px]">{[f.area, f.bedrooms != null ? `${f.bedrooms} bed` : null].filter(Boolean).join(" · ")}</p>
                <div className="mt-auto flex items-center justify-between pt-2 text-[12px]">
                  <span className="muted">{f.source ? SOURCE_LABELS[f.source] ?? f.source : ""} · saved {timeAgo(f.created_at)}</span>
                  <span className="flex gap-3">
                    <a href={f.url} target="_blank" rel="noopener noreferrer" className="font-semibold hover:underline" style={{ color: "var(--primary)" }}>Open ↗</a>
                    <button type="button" className="font-semibold" style={{ color: "var(--danger)" }} onClick={() => delFav.mutate(f.id)}>Remove</button>
                  </span>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
