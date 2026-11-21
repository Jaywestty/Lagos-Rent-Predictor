import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getAreas } from "../services/api";
import { nairaShort } from "../lib/format";
import { MoveInCalculator } from "../components/rent/MoveInCalculator";

export function MoveIn() {
  const [params, setParams] = useSearchParams();
  const area = params.get("area");
  const rentParam = Number(params.get("rent") || 0) || null;
  const areas = useQuery({ queryKey: ["areas"], queryFn: getAreas, staleTime: 5 * 60_000 });
  const fromArea = areas.data?.find((a) => a.area === area)?.median_ngn ?? null;

  useEffect(() => { document.title = "Move-in cost · Lagos Rent"; }, []);

  return (
    <div className="mx-auto w-full max-w-[1080px] px-4 pb-12 pt-6 md:px-10 md:pt-10">
      <h1 className="text-[28px] font-bold">Total move-in cost</h1>
      <p className="muted mt-1 max-w-[640px] text-[15px]">In Lagos the first payment is far more than the rent. Agency, legal and caution fees, plus paying a year or two upfront, add up fast. Start from the typical rent in an area, or type your own.</p>
      <div className="mt-5 flex flex-wrap items-center gap-2">
        <label className="text-[13px] font-semibold" htmlFor="area-pick">Start from an area</label>
        <select id="area-pick" className="input" value={area ?? ""} onChange={(e) => { const p = new URLSearchParams(); if (e.target.value) p.set("area", e.target.value); setParams(p, { replace: true }); }}>
          <option value="">Type my own rent</option>
          {(areas.data ?? []).map((a) => <option key={a.area} value={a.area}>{a.area} · about {nairaShort(a.median_ngn)} a year</option>)}
        </select>
      </div>
      <div className="mt-6">
        <MoveInCalculator key={area ?? "custom"} initialRent={rentParam ?? fromArea} area={area} />
      </div>
    </div>
  );
}
