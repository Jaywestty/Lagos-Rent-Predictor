export function LoadingState({ text = "Checking listings..." }: { text?: string }) {
  return (
    <div className="card fade-in p-4 md:p-6" aria-label="Loading" role="status">
      <p key={text} className="fade-in text-[14px] font-semibold">{text}</p>
      <div className="skeleton mt-4 h-10 w-1/2" />
      <div className="skeleton mt-3 h-4" />
      <div className="skeleton mt-2 h-4 w-5/6" />
      <div className="skeleton mt-2 h-4 w-2/3" />
    </div>
  );
}

export function NoDataState({ suggestions = [], onPick }: { suggestions?: string[]; onPick?: (area: string) => void }) {
  return (
    <section className="card rise p-6" aria-label="No data">
      <h2 className="text-[17px] font-semibold">No recent listings for that search</h2>
      <p className="muted mt-1 text-[15px]">We only answer from real listings, and none match yet. Try a nearby area or a different size.</p>
      {suggestions.length > 0 && onPick && (
        <div className="mt-3 flex flex-wrap gap-2">
          {suggestions.map((a) => <button key={a} type="button" className="chip" onClick={() => onPick(a)}>Try {a}</button>)}
        </div>
      )}
    </section>
  );
}

const IN_SCOPE_EXAMPLES = ["2 bedroom rent in Yaba", "Self contain under ₦1M", "Lekki or Ajah for a 3 bedroom?", "Is agent fee negotiable?"];

export function OutOfScopeState({ message, onAsk }: { message?: string; onAsk?: (q: string) => void }) {
  return (
    <section className="card rise p-6" aria-label="Out of scope">
      <h2 className="text-[18px] font-bold">That is outside what I can help with</h2>
      <p className="muted mt-1 text-[15px] leading-relaxed">{message || "I only answer questions about renting in Lagos: prices, budgets, areas and renting advice."}</p>
      {onAsk && (
        <div className="mt-4 flex flex-wrap gap-2">
          {IN_SCOPE_EXAMPLES.map((q) => <button key={q} type="button" className="chip" onClick={() => onAsk(q)}>{q}</button>)}
        </div>
      )}
    </section>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <section className="card rise p-6" aria-label="Error"><h2 className="text-[17px] font-semibold">Something went wrong</h2><p className="muted mt-1 text-[15px]">{message}</p><button type="button" onClick={onRetry} className="btn mt-3">Retry</button></section>;
}

export function PageError({ message }: { message: string }) {
  return <section className="card rise p-6" role="alert"><h2 className="text-[17px] font-semibold">Could not load this page</h2><p className="muted mt-1 text-[15px]">{message}</p></section>;
}

export function Skeleton({ rows = 3 }: { rows?: number }) {
  return <div className="space-y-3" role="status" aria-label="Loading">{Array.from({ length: rows }, (_, i) => <div key={i} className="skeleton h-20" />)}</div>;
}
