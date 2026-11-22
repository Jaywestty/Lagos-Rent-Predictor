import type { AskResult } from "../../services/api";
import { RentAnswer } from "../rent/RentAnswer";
import { ListingRows } from "../rent/ListingRows";
import { AdviceAnswer } from "../rent/AdviceAnswer";
import { ComparisonAnswer } from "../rent/ComparisonAnswer";
import { ErrorState, NoDataState, OutOfScopeState } from "../states/States";

type Props = {
  result: AskResult;
  question: string;
  onAsk: (q: string) => void;
  onRetry?: () => void;
  suggestions: string[];
};

/** One answer in a chat: rent card and listings, advice, or the right empty/error state. */
export function AnswerView({ result, question, onAsk, onRetry, suggestions }: Props) {
  const { answer, listings, advice } = result;
  if (advice) return <AdviceAnswer advice={advice} />;
  if (!answer) return null;
  if (answer.comparison?.options?.length && answer.status !== "error" && answer.status !== "out_of_scope") {
    return <ComparisonAnswer answer={answer} />;
  }
  if (answer.status === "success" || answer.status === "low_data") {
    return (
      <div className="space-y-4">
        {answer.status === "low_data" && <p className="fade-in rounded-xl px-4 py-3 text-[14px]" role="note" style={{ background: "var(--sand)" }}><strong>Very few listings.</strong> Treat this as a rough guide, not a price to hold an agent to.</p>}
        <RentAnswer answer={answer} question={question} onAsk={onAsk} />
        <ListingRows listings={listings.slice(0, 9)} typical={answer.median} />
      </div>
    );
  }
  if (answer.status === "no_data") return <NoDataState help={answer.no_data} question={question} suggestions={suggestions} onAsk={onAsk} />;
  if (answer.status === "out_of_scope") return <OutOfScopeState message={answer.message} onAsk={onAsk} />;
  if (answer.status === "clarification") {
    return (
      <section className="card rise p-6" aria-label="Clarification">
        <h3 className="text-[17px] font-semibold">{answer.clarification?.question || "Which area?"}</h3>
        <div className="mt-3 flex flex-wrap gap-2">
          {(answer.clarification?.options?.length ? answer.clarification.options : suggestions.slice(0, 4)).map((opt) => (
            <button key={opt} type="button" onClick={() => onAsk(`${opt} rent`)} className="chip">{opt}</button>
          ))}
        </div>
      </section>
    );
  }
  return <ErrorState message={answer.message || "Something failed on our side."} onRetry={onRetry ?? (() => onAsk(question))} />;
}
