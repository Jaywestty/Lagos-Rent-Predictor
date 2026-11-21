import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getAreas } from "../services/api";
import { nairaShort, timeAgo } from "../lib/format";
import { PageError, Skeleton } from "../components/states/States";

export function Areas() {
  const q = useQuery({ queryKey: ["areas"], queryFn: getAreas, staleTime: 5 * 60_000 });
  const [filter, setFilter] = useState("");
  const items = useMemo(() => (q.data ?? []).filter((a) => a.area.toLowerCase().includes(filter.trim().toLowerCase())), [q.data, filter]);
  const maxMedian = Math.max(...(q.data ?? []).map((a) => a.median_ngn ?? 0), 1);

  return (
    <div className="mx-auto w-full max-w-[1080px] px-4 pb-12 pt-6 md:px-10 md:pt-10">
      <div className="stagger">
        <h1 className="text-[28px] font-bold">Area guides</h1>
        <p className="muted mt-1 max-w-[640px] text-[15px]">What it is like to live in each area, how much rent really is, and how prices are moving. Rent figures come from listings; the area notes come from cited web sources.</p>
      </div>
      <input className="input mt-5 w-full max-w-[360px]" placeholder="Find an area" aria-label="Find an area" value={filter} onChange={(e) => setFilter(e.target.value)} />
      <div className="mt-5">
        {q.isPending && <Skeleton rows={4} />}
        {q.isError && <PageError message="The area list did not load. Check your connection and refresh." />}
        {q.data && !items.length && <p className="muted">No area matches “{filter}”.</p>}
        <div className="stagger grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((a) => (
            <Link key={a.area} to={`/areas/${encodeURIComponent(a.area)}`} className="card lift block p-4">
              <div className="flex items-baseline justify-between gap-2">
                <h2 className="text-[17px] font-semibold">{a.area}</h2>
                <span className="muted text-[12px]">{a.listing_count} listing{a.listing_count === 1 ? "" : "s"}</span>
              </div>
              <p className="tabular mt-2 text-[22px] font-bold">{nairaShort(a.median_ngn)}<span className="muted text-[12px] font-normal"> median /yr</span></p>
              <div className="mt-2 h-[6px] rounded" style={{ background: "var(--soft)" }} aria-hidden>
                <div className="grow-x h-[6px] rounded" style={{ width: `${((a.median_ngn ?? 0) / maxMedian) * 100}%`, background: "var(--primary)" }} />
              </div>
              <p className="muted mt-2 text-[12px]">Newest listing {timeAgo(a.latest_listing_at)}</p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
