import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { ask, type AskResult, getAreas, getThread, newRequestKey, newThreadId, track } from "../../services/api";
import { useChrome } from "../../lib/useSiteStats";
import { LoadingState } from "../states/States";
import { AnswerView } from "./AnswerView";
import { ChatHistory, ChatList, useThreads } from "./ChatHistory";
import { Welcome } from "./Welcome";

const PLACEHOLDERS = ["Try: rent in Gbagada", "Try: 2 bedroom in Yaba", "Try: self contain in Lekki", "Try: rent in Ikoyi"];
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

type Exchange = { key: string; question: string; state: "pending" | "done" | "error"; result?: AskResult; limited?: string };

function PlusIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden><path d="M12 5v14M5 12h14" /></svg>;
}

export function Chat() {
  const qc = useQueryClient();
  const { setChatActive } = useChrome();
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
  const threads = useThreads();

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

  const inChat = exchanges.length > 0 || Boolean(urlThread);
  useEffect(() => { setChatActive(inChat); return () => setChatActive(false); }, [inChat, setChatActive]);

  const run = useCallback(async (key: string, text: string, tid: string) => {
    try {
      const result = await ask(text, tid, newRequestKey());
      setExchanges((xs) => xs.map((x) => (x.key === key ? { ...x, state: "done", result } : x)));
      // The first real answer creates the chat on the server (errors are not saved): put it in the
      // address so refresh or history reopens it.
      if (result.answer && !["error", "out_of_scope", "greeting"].includes(result.answer.status)) {
        if (!params.get("c")) setParams({ c: tid }, { replace: true });
        qc.invalidateQueries({ queryKey: ["threads"] });
      }
    } catch (err) {
      // 429 = usage limit: show the server's "slow down" message instead of blaming the connection.
      const limited = (err as { status?: number })?.status === 429 ? (err as Error).message : undefined;
      setExchanges((xs) => xs.map((x) => (x.key === key ? { ...x, state: "error", limited } : x)));
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
  const title = exchanges[0]?.question || saved.data?.title || "New chat";

  const composer = (
    <form className="flex items-center gap-2 rounded-2xl border p-1.5 shadow-[0_8px_24px_rgba(27,36,32,.06)]" style={{ background: "var(--surface)", borderColor: "var(--border)" }}
      onSubmit={(e) => { e.preventDefault(); submit(question); }} aria-label="Ask about rent">
      <input
        aria-label="Your question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder={inChat ? "Ask a follow-up" : PLACEHOLDERS[placeholderIdx % PLACEHOLDERS.length]}
        onFocus={() => setPlaceholderIdx((i) => i + 1)}
        className="h-12 min-w-0 flex-1 bg-transparent px-3 text-[16px] outline-none md:h-14"
        style={{ color: "var(--text)" }}
      />
      <button type="submit" className="btn h-12 min-w-[76px] md:h-14 md:min-w-[110px]" disabled={pending}>{pending ? "…" : "Ask"}</button>
    </form>
  );

  const quickSearch = (
    <div className="space-y-3" aria-label="Quick search">
      {[
        { label: "Area", items: areaChips.map((a) => ({ key: a, text: a, on: area === a, click: () => pick(area, a, setArea, "area") })) },
        { label: "Size", items: BEDS.map((b) => ({ key: b.text, text: b.label, on: beds === b.text, click: () => pick(beds, b.text, setBeds, "beds") })) },
        { label: "Budget", items: BUDGETS.map((b) => ({ key: String(b.value), text: b.label, on: budget === b.value, click: () => pick(budget, b.value, setBudget, "budget") })) },
      ].map((row) => (
        <div key={row.label} className="flex flex-wrap items-center gap-2" role="group" aria-label={row.label}>
          <span className="kicker w-16 text-[11px]">{row.label}</span>
          {row.items.map((i) => <button key={i.key} type="button" className="chip" aria-pressed={i.on} onClick={i.click}>{i.text}</button>)}
        </div>
      ))}
      {hasFilters && (
        <div className="fade-in flex flex-wrap items-center gap-2 sm:pl-[72px]">
          <button type="button" className="btn" onClick={() => submit(buildQuestion(area, beds, budget))}>Search: {buildQuestion(area, beds, budget)}</button>
          <button type="button" className="btn-ghost" onClick={() => { setArea(null); setBeds(null); setBudget(null); }}>Clear</button>
        </div>
      )}
    </div>
  );

  const historyDrawer = <ChatHistory open={historyOpen} onClose={() => setHistoryOpen(false)} activeId={inChat ? threadId : null} onOpen={openChat} onNew={newChat} />;

  if (!inChat) {
    return (
      <>
        {historyDrawer}
        <Welcome composer={composer} quickSearch={quickSearch} onAsk={submit} chatCount={threads.data?.length ?? 0} onOpenChats={() => setHistoryOpen(true)} />
      </>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-[1200px]" style={{ minHeight: "calc(100vh - 64px)" }}>
      {historyDrawer}
      {/* Desktop: chats stay in view. */}
      <aside aria-label="Your chats" className="sticky top-[64px] hidden h-[calc(100vh-64px)] w-[280px] shrink-0 flex-col gap-3 border-r px-3 py-5 lg:flex" style={{ borderColor: "var(--border)" }}>
        <button type="button" className="btn inline-flex items-center justify-center gap-2" onClick={newChat}><PlusIcon />New chat</button>
        <ChatList activeId={threadId} onOpen={openChat} onNew={newChat} />
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Phone/tablet: a compact bar with the chat title. */}
        <div className="sticky top-[64px] z-10 flex items-center gap-2 border-b px-3 py-2 backdrop-blur lg:hidden" style={{ borderColor: "var(--border)", background: "color-mix(in srgb, var(--bg) 92%, transparent)" }}>
          <button type="button" className="btn-ghost grid w-11 place-items-center px-0" aria-label="Your chats" aria-haspopup="dialog" onClick={() => setHistoryOpen(true)}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M4 6h16M4 12h16M4 18h10" /></svg>
          </button>
          <p className="font-display min-w-0 flex-1 truncate text-center text-[15px] font-bold">{title}</p>
          <button type="button" className="btn grid w-11 place-items-center px-0" aria-label="New chat" onClick={newChat}><PlusIcon /></button>
        </div>

        <div className="mx-auto w-full max-w-[860px] flex-1 px-4 pb-4 pt-6 md:px-8">
          {saved.isPending && urlThread && !exchanges.length && <LoadingState text="Opening your chat..." />}
          {saved.isError && !exchanges.length && (
            <section className="card p-6" role="alert">
              <h2 className="text-[19px] font-bold">Chat not found</h2>
              <p className="muted mt-1 text-[14px]">It may have been deleted, or it was started on another device.</p>
              <button type="button" className="btn mt-3" onClick={newChat}>Start a new chat</button>
            </section>
          )}
          <ol className="space-y-10" aria-live="polite">
            {exchanges.map((x) => (
              <li key={x.key} className="space-y-4">
                <div className="flex justify-end">
                  <p className="fade-in max-w-[85%] rounded-2xl rounded-br-md px-4 py-2.5 text-[15px]" style={{ background: "var(--soft)" }}>{x.question}</p>
                </div>
                {x.state === "pending" && <LoadingState text={stages[stage]} />}
                {x.state === "error" && (
                  <section className="card p-5" role="alert">
                    <p className="text-[15px] font-semibold">{x.limited ? "Easy, one moment." : "That did not go through."}</p>
                    <p className="muted mt-1 text-[14px]">{x.limited || "Check your connection. Your question is kept."}</p>
                    <button type="button" className="btn mt-3" onClick={() => retry(x)}>Try again</button>
                  </section>
                )}
                {x.state === "done" && x.result && <AnswerView result={x.result} question={x.question} onAsk={submit} onRetry={() => retry(x)} suggestions={areaChips} />}
              </li>
            ))}
          </ol>
          <div ref={bottom} />
        </div>

        <div className="sticky bottom-0 z-10 px-4 pb-4 pt-3 md:px-8" style={{ background: "linear-gradient(180deg, transparent, var(--bg) 35%)", paddingBottom: "max(16px, env(safe-area-inset-bottom))" }}>
          <div className="mx-auto max-w-[860px]">{composer}</div>
        </div>
      </div>
    </div>
  );
}
