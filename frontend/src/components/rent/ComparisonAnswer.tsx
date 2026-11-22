import { Link } from "react-router-dom";
import type { ChatAnswer } from "../../types/api";
import { nairaShort } from "../../lib/format";
import { ConfidenceBadge } from "./RentAnswer";

/** A comparison question ("Yaba or Surulere for a 2 bedroom?") answered option by option from listings. */
export function ComparisonAnswer({ answer }: { answer: ChatAnswer }) {
  const options = answer.comparison?.options ?? [];
  const priced = options.filter((o) => o.median != null);
  const cheapest = priced.length > 1 ? priced.reduce((a, b) => (a.median! <= b.median! ? a : b)) : null;
  const max = Math.max(...priced.map((o) => o.max ?? o.median ?? 0), 1);
  const areas = options.map((o) => o.area).filter((a): a is string => Boolean(a));
  const beds = options.find((o) => o.beds)?.beds;

  return (
    <section className="card rise overflow-hidden" aria-label="Comparison">
      <div className="space-y-5 p-5 md:p-7">
        <p className="kicker">Side by side · typical rent per year</p>
        {cheapest && (
          <p className="font-display text-[22px] font-bold leading-snug md:text-[26px]">
            {cheapest.label} is cheaper, at about <span className="figure" style={{ color: "var(--action)" }}>{nairaShort(cheapest.median)}</span> a year.
          </p>
        )}
        {!priced.length && <p className="text-[16px]">None of these options has recent listings yet, so there is nothing to compare.</p>}
        <ul className="space-y-4">
          {options.map((o) => (
            <li key={o.label} className="space-y-2">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <p className="text-[15px] font-semibold">{o.label}</p>
                <p className="figure text-[22px] font-bold">{o.median != null ? nairaShort(o.median) : "No data"}</p>
              </div>
              {o.median != null && (
                <div className="relative h-2 rounded-full" style={{ background: "var(--line)" }} aria-hidden>
                  <div className="grow-x absolute inset-y-0 left-0 rounded-full" style={{ width: `${((o.median ?? 0) / max) * 100}%`, background: o === cheapest ? "var(--action)" : "var(--secondary)" }} />
                </div>
              )}
              <div className="flex flex-wrap items-center gap-2 text-[12px]">
                <ConfidenceBadge confidence={o.confidence} count={o.count} />
                {o.min != null && o.max != null && <span className="muted">Range {nairaShort(o.min)} – {nairaShort(o.max)}</span>}
              </div>
              {o.caveat && <p className="muted text-[12px]">{o.caveat}</p>}
            </li>
          ))}
        </ul>
        {areas.length >= 2 && (
          <Link to={`/compare?areas=${encodeURIComponent(areas.join(","))}${beds ? `&beds=${beds}` : ""}`} className="btn-ghost inline-flex items-center">
            Open the full comparison →
          </Link>
        )}
      </div>
    </section>
  );
}
