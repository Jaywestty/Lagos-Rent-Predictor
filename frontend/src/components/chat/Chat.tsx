import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ask, newRequestKey } from "../../services/api";
import type { ChatAnswer } from "../../types/api";
import { RentAnswer } from "../rent/RentAnswer";
import { ErrorState, LoadingState, NoDataState, OutOfScopeState } from "../states/States";

const EXAMPLES = ["How much is rent in Gbagada?", "What about a 2 bedroom in Yaba?", "Rent in Ikeja for 3 bedroom", "How much is a self contain in Surulere?"];
const PLACEHOLDERS = ["Try: How much is rent in Gbagada?", "Try: 2 bedroom in Yaba", "Try: self contain in Lekki", "Try: rent in Ikoyi"];

export function Chat() {
  const [question, setQuestion] = useState("");
  const [sessionId] = useState<string | null>(() => localStorage.getItem("lagos-session"));
  const [requestKey, setRequestKey] = useState(newRequestKey());
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [submitted, setSubmitted] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["ask", submitted, requestKey],
    queryFn: () => ask(submitted!, sessionId, requestKey),
    enabled: Boolean(submitted),
    retry: false,
  });

  function submit(q: string) {
    const text = q.trim();
    if (!text) return;
    if (!sessionId) localStorage.setItem("lagos-session", crypto.randomUUID());
    setRequestKey(newRequestKey());
    setSubmitted(text);
    setQuestion("");
  }

  function retry() {
    setRequestKey(newRequestKey());
  }

  const answer = query.data as ChatAnswer | undefined;
  return (
    <div className="mx-auto w-full max-w-[760px] px-4 md:px-10">
      <section className="pt-6 md:pt-10">
        <h1 className="text-[28px] font-bold leading-tight">Rent in Lagos, priced from real listings.</h1>
        <p className="mt-2 text-[15px]" style={{ color: "var(--secondary)" }}>Ask like you talk. We show the range, the median and how many listings back it.</p>
        <div className="mt-4 flex flex-wrap gap-2" aria-label="Example questions">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => submit(ex)} className="min-h-[44px] rounded-full border px-3 py-2 text-[13px] font-semibold" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>{ex}</button>
          ))}
        </div>
      </section>
      <form
        className="card mt-4 flex items-center gap-2 p-2"
        onSubmit={(e) => { e.preventDefault(); submit(question); }}
        aria-label="Ask about rent"
      >
        <input
          aria-label="Your question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={PLACEHOLDERS[placeholderIdx % PLACEHOLDERS.length]}
          onFocus={() => setPlaceholderIdx((i) => i + 1)}
          className="h-[52px] min-w-0 flex-1 rounded bg-transparent px-3 text-[15px] outline-none md:h-14"
          style={{ color: "var(--text)" }}
        />
        <button type="submit" className="min-h-[48px] rounded px-4 text-[14px] font-semibold text-white md:min-w-[120px]" style={{ background: "var(--primary)" }}>Ask</button>
      </form>
      <button type="button" onClick={() => setPlaceholderIdx((i) => i + 1)} className="sr-only">Cycle placeholder</button>
      <div className="mt-4 space-y-4 pb-10">
        {query.isPending && submitted && <LoadingState />}
        {query.isError && <ErrorState message="Network error. Your question is preserved." onRetry={retry} />}
        {answer && answer.status !== "no_data" && answer.status !== "out_of_scope" && answer.status !== "clarification" && answer.status !== "error" && <RentAnswer answer={answer} />}
        {answer?.status === "no_data" && <NoDataState />}
        {answer?.status === "out_of_scope" && <OutOfScopeState />}
        {answer?.status === "clarification" && (
          <section className="card p-6" aria-label="Clarification">
            <h2 className="text-[17px] font-semibold">{answer.clarification?.question || "Which area?"}</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              {answer.clarification?.options?.map((opt) => (
                <button key={opt} type="button" onClick={() => submit(`${opt} rent`)} className="min-h-[44px] rounded border px-3 text-[14px] font-semibold" style={{ borderColor: "var(--border)" }}>{opt}</button>
              ))}
            </div>
          </section>
        )}
        {answer && answer.status === "low_data" && <p className="card p-4 text-[15px]" role="note">Low data: this estimate is based on very few listings. Treat it as a rough guide.</p>}
      </div>
    </div>
  );
}
