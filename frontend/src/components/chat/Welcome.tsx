import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { nairaShort, timeAgo } from "../../lib/format";
import { useSiteStats } from "../../lib/useSiteStats";

const HEADLINE = ["Know", "what", "rent", "really", "costs"];
const HEADLINE_2 = ["before", "you", "call", "an", "agent."];
const EXAMPLES = ["2 bedroom in Yaba", "Self contain in Surulere", "Rent in Lekki under ₦5M", "Is Gbagada prone to flooding?"];

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** Counts up to a number once it scrolls into view (instantly under reduced motion). */
function CountUp({ value, format = (n: number) => n.toLocaleString("en-NG") }: { value: number; format?: (n: number) => string }) {
  const [shown, setShown] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!value) return;
    if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") { setShown(value); return; }
    let raf = 0;
    const io = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      io.disconnect();
      const start = performance.now();
      const tick = (t: number) => {
        const p = Math.min(1, (t - start) / 1100);
        setShown(Math.round(value * (1 - Math.pow(1 - p, 3))));
        if (p < 1) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    });
    if (ref.current) io.observe(ref.current);
    return () => { io.disconnect(); cancelAnimationFrame(raf); };
  }, [value]);
  return <span ref={ref} className="tabular">{format(shown)}</span>;
}

type Props = { composer: React.ReactNode; quickSearch: React.ReactNode; onAsk: (q: string) => void; chatCount?: number; onOpenChats: () => void };

export function Welcome({ composer, quickSearch, onAsk, chatCount = 0, onOpenChats }: Props) {
  const stats = useSiteStats();
  const s = stats.data;
  const ticker = s?.top ?? [];

  return (
    <div>
      {/* Hero */}
      <section className="ledger relative overflow-hidden border-b" style={{ borderColor: "var(--border)" }}>
        <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: "linear-gradient(180deg, color-mix(in srgb, var(--bg) 20%, transparent), var(--bg) 92%)" }} />
        <div className="relative mx-auto max-w-[1200px] px-4 pb-10 pt-10 md:px-8 md:pb-14 md:pt-16">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="kicker fade-in">Lagos rent · from this week&apos;s listings</p>
            {chatCount > 0 && (
              <button type="button" className="btn-ghost fade-in" onClick={onOpenChats}>Your chats · {chatCount}</button>
            )}
          </div>
          <h1 className="mt-4 max-w-[900px] text-[40px] font-bold leading-[1.02] sm:text-[56px] md:text-[72px]">
            <span className="block">{HEADLINE.map((w, i) => <span key={w}><span className="word-up" style={{ animationDelay: `${i * 70}ms` }}>{w}</span>{" "}</span>)}</span>
            <span className="block" style={{ color: "var(--action)" }}>{HEADLINE_2.map((w, i) => <span key={w}><span className="word-up" style={{ animationDelay: `${(i + HEADLINE.length) * 70}ms` }}>{w}</span>{" "}</span>)}</span>
          </h1>
          <p className="fade-in mt-5 max-w-[580px] text-[17px] leading-relaxed md:text-[19px]" style={{ animationDelay: "700ms", color: "var(--secondary)" }}>
            Ask about any area in Lagos. You get the typical yearly rent, the range most listings fall in, and the listings behind it.
          </p>
          <div className="rise mt-7 max-w-[760px]" style={{ animationDelay: "820ms" }}>{composer}</div>
          <div className="fade-in mt-4 flex max-w-[760px] flex-wrap gap-2" style={{ animationDelay: "950ms" }} aria-label="Example questions">
            {EXAMPLES.map((q) => <button key={q} type="button" className="chip" onClick={() => onAsk(q)}>{q}</button>)}
          </div>
          <details className="fade-in mt-5 max-w-[760px]" style={{ animationDelay: "1050ms" }}>
            <summary className="cursor-pointer text-[14px] font-semibold" style={{ color: "var(--action)" }}>Or build a search by area, size and budget</summary>
            <div className="mt-4">{quickSearch}</div>
          </details>
        </div>
      </section>

      {/* Live ticker of this week's typical rents */}
      {ticker.length > 0 && (
        <section aria-label="Typical rent this week" className="ticker overflow-hidden border-b py-4" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <div className="ticker-track gap-3 px-4">
            {[...ticker, ...ticker].map((a, i) => (
              <button key={`${a.area}-${i}`} type="button" tabIndex={i < ticker.length ? 0 : -1} aria-hidden={i >= ticker.length}
                onClick={() => onAsk(`How much is rent in ${a.area}?`)}
                className="flex shrink-0 items-baseline gap-2 rounded-full border px-4 py-2 text-[14px] transition-colors hover:border-[color:var(--action)]" style={{ borderColor: "var(--border)" }}>
                <span className="font-semibold">{a.area}</span>
                <span className="figure font-bold" style={{ color: "var(--action)" }}>{nairaShort(a.median_ngn)}</span>
                <span className="muted text-[12px]">{a.listing_count} listings</span>
              </button>
            ))}
          </div>
        </section>
      )}

      <div className="mx-auto max-w-[1200px] px-4 md:px-8">
        {/* Numbers */}
        {s && s.listings > 0 && (
          <section aria-label="The data" className="grid gap-px overflow-hidden rounded-2xl border sm:grid-cols-3" style={{ borderColor: "var(--border)", background: "var(--border)", marginTop: 40 }}>
            {[
              { label: "listings from the last few weeks", value: <CountUp value={s.listings} /> },
              { label: "Lagos areas covered", value: <CountUp value={s.areas} /> },
              { label: "since the newest listing", value: <span>{timeAgo(s.newest)}</span> },
            ].map((item) => (
              <div key={item.label} className="p-6" style={{ background: "var(--surface)" }}>
                <p className="figure text-[36px] font-bold leading-none md:text-[44px]">{item.value}</p>
                <p className="muted mt-2 text-[14px]">{item.label}</p>
              </div>
            ))}
          </section>
        )}

        {/* How it works */}
        <section aria-labelledby="how" className="mt-14">
          <h2 id="how" className="text-[28px] font-bold md:text-[34px]">How it works</h2>
          <ol className="mt-6 grid gap-6 md:grid-cols-3">
            {[
              { n: "01", t: "Ask in plain words", d: "Type it like you would ask a friend: the area, the size, your budget.", to: "/", cta: "Ask now" },
              { n: "02", t: "See the going rate", d: "Typical rent, the range, how many listings back it, and the listings themselves.", to: "/areas", cta: "Browse areas" },
              { n: "03", t: "Count the full cost", d: "Agency, legal and caution fees can add a lot to the first payment. Add them up first.", to: "/move-in", cta: "Move-in cost" },
            ].map((step, i) => (
              <li key={step.n} className="relative pt-5">
                <span aria-hidden className="draw-line absolute inset-x-0 top-0 h-[2px]" style={{ background: i === 0 ? "var(--action)" : "var(--border)", animationDelay: `${i * 150}ms` }} />
                <p className="figure text-[14px] font-bold" style={{ color: "var(--action)" }}>{step.n}</p>
                <h3 className="font-display mt-2 text-[21px] font-bold">{step.t}</h3>
                <p className="muted mt-2 text-[15px] leading-relaxed">{step.d}</p>
                <Link to={step.to} className="mt-3 inline-block text-[14px] font-semibold underline-offset-4 hover:underline" style={{ color: "var(--action)" }}>{step.cta} →</Link>
              </li>
            ))}
          </ol>
        </section>

        {/* Popular areas */}
        {ticker.length > 0 && (
          <section aria-labelledby="areas" className="mt-14">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <h2 id="areas" className="text-[28px] font-bold md:text-[34px]">Most-listed areas this week</h2>
              <Link to="/areas" className="text-[14px] font-semibold underline-offset-4 hover:underline" style={{ color: "var(--action)" }}>All area guides →</Link>
            </div>
            <div className="stagger mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {ticker.slice(0, 4).map((a) => (
                <Link key={a.area} to={`/areas/${encodeURIComponent(a.area)}`} className="card lift block p-5">
                  <p className="font-display text-[19px] font-bold">{a.area}</p>
                  <p className="figure mt-3 text-[30px] font-bold leading-none">{nairaShort(a.median_ngn)}</p>
                  <p className="muted mt-1 text-[13px]">typical rent a year · {a.listing_count} listings</p>
                  <p className="mt-4 text-[13px] font-semibold" style={{ color: "var(--action)" }}>Read the area guide →</p>
                </Link>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
