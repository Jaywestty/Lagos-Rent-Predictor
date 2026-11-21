import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { ask, getAreas, newRequestKey, track } from "../../services/api";
import { RentAnswer } from "../rent/RentAnswer";
import { ListingGrid } from "../rent/ListingCard";
import { AdviceAnswer } from "../rent/AdviceAnswer";
import { ErrorState, LoadingState, NoDataState, OutOfScopeState } from "../states/States";

const PLACEHOLDERS = ["Try: How much is rent in Gbagada?", "Try: 2 bedroom in Yaba", "Try: self contain in Lekki", "Try: rent in Ikoyi"];
const BEDS = [
  { label: "Self contain", text: "self contain" },
  { label: "1 bed", text: "1 bedroom" },
  { label: "2 bed", text: "2 bedroom" },
  { label: "3 bed", text: "3 bedroom" },
];
const BUDGETS = [
  { label: "Under ₦1M", value: 1_000_000 },
  { label: "Under ₦2.5M", value: 2_500_000 },
  { label: "Under ₦5M", value: 5_000_000 },
  { label: "Under ₦10M", value: 10_000_000 },
];
const FALLBACK_AREAS = ["Lekki", "Yaba", "Ikeja", "Gbagada", "Surulere", "Ajah"];

function buildQuestion(area: string | null, beds: string | null, budget: number | null): string {
  const what = beds || "rent";
  const where = area ? ` in ${area}` : " in Lagos";
  if (budget) return `${beds || "Places"} for rent${where} under ${budget.toLocaleString("en-NG")} naira a year`;
  return beds ? `${what} rent${where}` : `How much is rent${where}?`;
}

export function Chat() {
  const [question, setQuestion] = useState("");
  const [sessionId] = useState<string | null>(() => {
    try { return localStorage.getItem("lagos-session"); } catch { return null; }
  });
  const [requestKey, setRequestKey] = useState(newRequestKey());
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [area, setArea] = useState<string | null>(null);
  const [beds, setBeds] = useState<string | null>(null);
  const [budget, setBudget] = useState<number | null>(null);
  const [stage, setStage] = useState(0);
  const [params, setParams] = useSearchParams();

  const areas = useQuery({ queryKey: ["areas"], queryFn: getAreas, staleTime: 5 * 60_000, retry: false });
  const areaChips = (areas.data?.map((a) => a.area) ?? FALLBACK_AREAS).slice(0, 8);

  const query = useQuery({
    queryKey: ["ask", submitted, requestKey],
    queryFn: () => ask(submitted!, sessionId, requestKey),
    enabled: Boolean(submitted),
    retry: false,
  });

  useEffect(() => {
    if (!query.isFetching) return;
    setStage(0);
    const timers = [setTimeout(() => setStage(1), 1200), setTimeout(() => setStage(2), 3000)];
    return () => timers.forEach(clearTimeout);
  }, [query.isFetching, requestKey]);

  useEffect(() => {
    const q = params.get("q");
    if (q) {
      submit(q);
      setParams({}, { replace: true });
    }
  }, []);

  function submit(q: string) {
    const text = q.trim();
    if (!text) return;
    try { if (!sessionId) localStorage.setItem("lagos-session", crypto.randomUUID()); } catch { /* private mode */ }
    setRequestKey(newRequestKey());
    setSubmitted(text);
    setQuestion("");
  }

  function pick<T>(current: T | null, next: T, set: (v: T | null) => void, kind: string) {
    const value = current === next ? null : next;
    set(value);
    track("chip_click", { kind });
  }

  const data = query.data;
  const answer = data?.answer;
  const hasFilters = Boolean(area || beds || budget);
  const stages = ["Reading your question...", `Checking ${area ? `${area} ` : ""}listings...`, "Working out the range and median..."];

  return (
    <div className="mx-auto w-full max-w-[1080px] px-4 md:px-10">
      <section className="stagger pt-6 md:pt-10">
        <h1 className="text-[28px] font-bold leading-tight md:text-[34px]">Rent in Lagos, priced from real listings.</h1>
        <p className="muted mt-2 max-w-[620px] text-[15px]">Ask like you talk. We show the range, the median and how many listings back it, all from listings posted in the last few weeks.</p>
      </section>

      <form className="card mt-5 flex items-center gap-2 p-2" onSubmit={(e) => { e.preventDefault(); submit(question); }} aria-label="Ask about rent">
        <input
          aria-label="Your question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={PLACEHOLDERS[placeholderIdx % PLACEHOLDERS.length]}
          onFocus={() => setPlaceholderIdx((i) => i + 1)}
          className="h-[52px] min-w-0 flex-1 rounded bg-transparent px-3 text-[15px] outline-none md:h-14"
          style={{ color: "var(--text)" }}
        />
        <button type="submit" className="btn md:min-w-[120px]">Ask</button>
      </form>

      <section className="mt-4 space-y-3" aria-label="Quick search">
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Area">
          <span className="muted w-16 text-[12px] font-semibold uppercase tracking-wide">Area</span>
          {areaChips.map((a) => <button key={a} type="button" className="chip" aria-pressed={area === a} onClick={() => pick(area, a, setArea, "area")}>{a}</button>)}
        </div>
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Size">
          <span className="muted w-16 text-[12px] font-semibold uppercase tracking-wide">Size</span>
          {BEDS.map((b) => <button key={b.text} type="button" className="chip" aria-pressed={beds === b.text} onClick={() => pick(beds, b.text, setBeds, "beds")}>{b.label}</button>)}
        </div>
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Budget">
          <span className="muted w-16 text-[12px] font-semibold uppercase tracking-wide">Budget</span>
          {BUDGETS.map((b) => <button key={b.value} type="button" className="chip" aria-pressed={budget === b.value} onClick={() => pick(budget, b.value, setBudget, "budget")}>{b.label}</button>)}
        </div>
        {hasFilters && (
          <div className="fade-in flex flex-wrap items-center gap-2 pl-0 md:pl-[72px]">
            <button type="button" className="btn" onClick={() => submit(buildQuestion(area, beds, budget))}>Search: {buildQuestion(area, beds, budget)}</button>
            <button type="button" className="btn-ghost" onClick={() => { setArea(null); setBeds(null); setBudget(null); }}>Clear</button>
          </div>
        )}
      </section>

      <div className="mt-6 space-y-4 pb-12" aria-live="polite">
        {submitted && (query.isFetching || answer) && <p className="muted text-[13px]">You asked: <span style={{ color: "var(--text)" }}>{submitted}</span></p>}
        {query.isFetching && <LoadingState text={stages[stage]} />}
        {query.isError && !query.isFetching && <ErrorState message="Network error. Your question is preserved." onRetry={() => setRequestKey(newRequestKey())} />}
        {!query.isFetching && data?.advice && <AdviceAnswer advice={data.advice} />}
        {!query.isFetching && answer && !data?.advice && (answer.status === "success" || answer.status === "low_data") && (
          <>
            {answer.status === "low_data" && <p className="card fade-in p-4 text-[14px]" role="note"><strong>Low data:</strong> this estimate rests on very few listings. Treat it as a rough guide.</p>}
            <RentAnswer answer={answer} question={submitted ?? undefined} onAsk={submit} />
            {data.listings.length > 0 && (
              <section aria-label="Matching listings" className="fade-in">
                <h2 className="mb-3 text-[17px] font-semibold">Listings behind this answer</h2>
                <ListingGrid listings={data.listings.slice(0, 9)} />
              </section>
            )}
          </>
        )}
        {!query.isFetching && answer?.status === "no_data" && <NoDataState suggestions={areaChips.slice(0, 4)} onPick={(a) => submit(`How much is rent in ${a}?`)} />}
        {!query.isFetching && answer?.status === "out_of_scope" && <OutOfScopeState />}
        {!query.isFetching && answer?.status === "error" && <ErrorState message={answer.message || "Something failed on our side."} onRetry={() => setRequestKey(newRequestKey())} />}
        {!query.isFetching && answer?.status === "clarification" && (
          <section className="card rise p-6" aria-label="Clarification">
            <h2 className="text-[17px] font-semibold">{answer.clarification?.question || "Which area?"}</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              {(answer.clarification?.options?.length ? answer.clarification.options : areaChips.slice(0, 4)).map((opt) => (
                <button key={opt} type="button" onClick={() => submit(`${opt} rent`)} className="chip">{opt}</button>
              ))}
            </div>
          </section>
        )}
        {!submitted && (
          <section className="stagger grid gap-3 pt-2 sm:grid-cols-3" aria-label="What you can do">
            {[
              { t: "Area guides", d: "Commute, flooding, power and safety, with sources.", href: "/areas" },
              { t: "Compare areas", d: "Side-by-side rent for up to three areas.", href: "/compare" },
              { t: "Move-in cost", d: "Rent plus agency, legal and caution fees.", href: "/move-in" },
            ].map((c) => (
              <Link key={c.href} to={c.href} className="card lift block p-4">
                <p className="text-[15px] font-semibold">{c.t} →</p>
                <p className="muted mt-1 text-[13px]">{c.d}</p>
              </Link>
            ))}
          </section>
        )}
      </div>
    </div>
  );
}
