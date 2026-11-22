import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { ask, type AskResult, getAreas, getThread, newRequestKey, newThreadId, track } from "../../services/api";
import { LoadingState } from "../states/States";
import { AnswerView } from "./AnswerView";
import { ChatHistory } from "./ChatHistory";

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

type Exchange = { key: string; question: string; state: "pending" | "done" | "error"; result?: AskResult };

export function Chat() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const urlThread = params.get("c");
  const [threadId, setThreadId] = useState<string>(() => urlThread || newThreadId());
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [question, setQuestion] = useState("");
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [area, setArea] = useState<string | null>(null);
  const [beds, setBeds] = useState<string | null>(null);
  const [budget, setBudget] = useState<number | null>(null);
  const [stage, setStage] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  const areas = useQuery({ queryKey: ["areas"], queryFn: getAreas, staleTime: 5 * 60_000, retry: false });
  const areaChips = (areas.data?.map((a) => a.area) ?? FALLBACK_AREAS).slice(0, 8);

  // Opening a saved chat (?c=<id>): load its questions and answers.
  const saved = useQuery({ queryKey: ["thread", urlThread], queryFn: () => getThread(urlThread!), enabled: Boolean(urlThread), retry: false });
  useEffect(() => {
    if (!urlThread) return;
    setThreadId(urlThread);
    if (saved.data) {
      setExchanges(saved.data.messages.map((m) => ({
        key: `saved-${m.id}`, question: m.question, state: "done",
        result: { answer: m.payload.answer, listings: m.payload.listings || [], advice: m.payload.advice ?? null },
      })));
    }
  }, [urlThread, saved.data]);

  const pending = exchanges.some((e) => e.state === "pending");
  useEffect(() => {
    if (!pending) return;
    setStage(0);
    const timers = [setTimeout(() => setStage(1), 1200), setTimeout(() => setStage(2), 3000)];
    return () => timers.forEach(clearTimeout);
  }, [pending]);

  useEffect(() => {
    if (exchanges.length) bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [exchanges.length, pending]);

  const run = useCallback(async (key: string, text: string, tid: string) => {
    try {
      const result = await ask(text, tid, newRequestKey());
      setExchanges((xs) => xs.map((x) => (x.key === key ? { ...x, state: "done", result } : x)));
      // The first real answer creates the chat on the server (errors are not saved): put it in the
      // address so refresh or history reopens it.
      if (result.answer?.status !== "error") {
        if (!params.get("c")) setParams({ c: tid }, { replace: true });
        qc.invalidateQueries({ queryKey: ["threads"] });
      }
    } catch {
      setExchanges((xs) => xs.map((x) => (x.key === key ? { ...x, state: "error" } : x)));
    }
  }, [params, setParams, qc]);

  const submit = useCallback((q: string) => {
    const text = q.trim();
    if (!text || pending) return;
    const key = newRequestKey();
    setExchanges((xs) => [...xs, { key, question: text, state: "pending" }]);
    setQuestion("");
    void run(key, text, threadId);
  }, [pending, run, threadId]);

  // ?q=<question> (from saved searches): ask it in a fresh chat.
  useEffect(() => {
    const q = params.get("q");
    if (q) {
      setParams({}, { replace: true });
      submit(q);
    }
  }, []);

  function retry(x: Exchange) {
    setExchanges((xs) => xs.map((e) => (e.key === x.key ? { ...e, state: "pending" } : e)));
    void run(x.key, x.question, threadId);
  }

  function newChat() {
    setExchanges([]);
    setThreadId(newThreadId());
    setParams({}, { replace: true });
  }

  function openChat(id: string) {
    if (id === threadId && exchanges.length) return;
    setExchanges([]);
    setParams({ c: id }, { replace: false });
  }

  function pick<T>(current: T | null, next: T, set: (v: T | null) => void, kind: string) {
    set(current === next ? null : next);
    track("chip_click", { kind });
  }

  const hasFilters = Boolean(area || beds || budget);
  const stages = ["Reading your question...", `Checking ${area ? `${area} ` : ""}listings...`, "Working out the price range..."];
  const inChat = exchanges.length > 0 || Boolean(urlThread);

  const composer = (
    <form className="card flex items-center gap-2 p-2" onSubmit={(e) => { e.preventDefault(); submit(question); }} aria-label="Ask about rent">
      <input
        aria-label="Your question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder={inChat ? "Ask a follow-up, e.g. what about a 3 bedroom?" : PLACEHOLDERS[placeholderIdx % PLACEHOLDERS.length]}
        onFocus={() => setPlaceholderIdx((i) => i + 1)}
        className="h-[52px] min-w-0 flex-1 rounded bg-transparent px-3 text-[15px] outline-none md:h-14"
        style={{ color: "var(--text)" }}
      />
      <button type="submit" className="btn md:min-w-[120px]" disabled={pending}>{pending ? "..." : "Ask"}</button>
    </form>
  );

  return (
    <div className="mx-auto w-full max-w-[1080px] px-4 md:px-10">
      <div className="sticky top-[60px] z-10 -mx-4 flex items-center justify-between gap-2 px-4 py-2 md:-mx-10 md:px-10" style={{ background: "color-mix(in srgb, var(--bg) 92%, transparent)", backdropFilter: "blur(6px)" }}>
        <button type="button" className="btn-ghost" onClick={() => setHistoryOpen(true)} aria-haspopup="dialog">☰ Your chats</button>
        {inChat && <button type="button" className="btn-ghost" onClick={newChat}>+ New chat</button>}
      </div>
      <ChatHistory open={historyOpen} onClose={() => setHistoryOpen(false)} activeId={inChat ? threadId : null} onOpen={openChat} onNew={newChat} />

      {!inChat && (
        <>
          <section className="stagger pt-4 md:pt-6">
            <h1 className="text-[28px] font-bold leading-tight md:text-[34px]">Rent in Lagos, priced from real listings.</h1>
            <p className="muted mt-2 max-w-[620px] text-[15px]">Ask like you talk. We show the typical rent, the price range and how many listings back it, all from listings posted in the last few weeks.</p>
          </section>
          <div className="mt-5">{composer}</div>
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
          <section className="stagger grid gap-3 pb-12 pt-8 sm:grid-cols-3" aria-label="What you can do">
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
        </>
      )}

      {inChat && (
        <div className="pb-4 pt-4">
          {saved.isPending && urlThread && !exchanges.length && <LoadingState text="Opening your chat..." />}
          {saved.isError && !exchanges.length && (
            <section className="card p-6" role="alert">
              <h2 className="text-[17px] font-semibold">Chat not found</h2>
              <p className="muted mt-1 text-[14px]">It may have been deleted, or it was started on another device.</p>
              <button type="button" className="btn mt-3" onClick={newChat}>Start a new chat</button>
            </section>
          )}
          <ol className="space-y-8" aria-live="polite">
            {exchanges.map((x) => (
              <li key={x.key} className="space-y-3">
                <div className="flex justify-end">
                  <p className="fade-in max-w-[85%] rounded-2xl rounded-br-md px-4 py-2 text-[15px]" style={{ background: "var(--soft)" }}>{x.question}</p>
                </div>
                {x.state === "pending" && <LoadingState text={stages[stage]} />}
                {x.state === "error" && (
                  <section className="card p-5" role="alert">
                    <p className="text-[15px] font-semibold">That did not go through.</p>
                    <p className="muted mt-1 text-[14px]">Check your connection. Your question is kept.</p>
                    <button type="button" className="btn mt-3" onClick={() => retry(x)}>Try again</button>
                  </section>
                )}
                {x.state === "done" && x.result && <AnswerView result={x.result} question={x.question} onAsk={submit} onRetry={() => retry(x)} suggestions={areaChips} />}
              </li>
            ))}
          </ol>
          <div ref={bottom} />
        </div>
      )}

      {inChat && (
        <div className="sticky bottom-0 z-10 -mx-4 px-4 pb-4 pt-2 md:-mx-10 md:px-10" style={{ background: "linear-gradient(180deg, transparent, var(--bg) 30%)" }}>
          {composer}
        </div>
      )}
    </div>
  );
}
