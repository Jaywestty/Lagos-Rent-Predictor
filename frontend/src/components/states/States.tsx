import { Mascot, MascotHead } from "../brand/Mascot";
import type { Greeting, NoDataHelp } from "../../types/api";

export function LoadingState({ text = "Checking listings..." }: { text?: string }) {
  return (
    <div className="card fade-in p-4 md:p-6" aria-label="Loading" role="status">
      <div className="flex items-center gap-3">
        <Mascot size={48} mood="think" />
        <p key={text} className="fade-in text-[14px] font-semibold">{text}</p>
      </div>
      <div className="skeleton mt-4 h-10 w-1/2" />
      <div className="skeleton mt-3 h-4" />
      <div className="skeleton mt-2 h-4 w-5/6" />
      <div className="skeleton mt-2 h-4 w-2/3" />
    </div>
  );
}

type NoDataProps = { help?: NoDataHelp | null; question?: string; suggestions?: string[]; onAsk?: (q: string) => void };

/** Explains why a search found nothing and offers the most useful next step. */
export function NoDataState({ help, question = "", suggestions = [], onAsk }: NoDataProps) {
  const area = help?.area;
  const others = (help?.suggestions?.length ? help.suggestions : suggestions).filter((a) => a !== area).slice(0, 4);
  // The same question with the misspelt area swapped for the suggestion (plain text search, no regex).
  const at = area ? question.toLowerCase().indexOf(area.toLowerCase()) : -1;
  const fixed = help?.did_you_mean && area
    ? (at >= 0 ? question.slice(0, at) + help.did_you_mean + question.slice(at + area.length) : `How much is rent in ${help.did_you_mean}?`)
    : null;

  let title = "No recent listings for that search";
  let body = "Every answer here comes from real listings posted in the last few weeks, and none match this yet. Try a nearby area or a different size.";
  if (help?.did_you_mean) {
    title = `Did you mean ${help.did_you_mean}?`;
    body = `We couldn't find an area called "${area}".`;
  } else if (area && help?.covered) {
    title = `Nothing of that kind in ${area} right now`;
    body = `We have ${help.area_listings ?? "some"} recent listings in ${area}, just none that match this size or type. Try a different size, or see everything in ${area}.`;
  } else if (area) {
    title = `We don't cover ${area} yet`;
    body = `There are no recent listings for ${area}, so we can't give a price without guessing. New listings are added every week, mostly from the busiest parts of Lagos. You can still ask what living there is like.`;
  }

  return (
    <section className="card rise p-6" aria-label="No data">
      <div className="flex items-start gap-3">
        <span className="shrink-0"><Mascot size={44} mood="idle" /></span>
        <div className="min-w-0">
          <h2 className="text-[18px] font-bold">{title}</h2>
          <p className="muted mt-1 text-[15px] leading-relaxed">{body}</p>
        </div>
      </div>
      {onAsk && (
        <div className="mt-4 flex flex-wrap gap-2">
          {fixed && <button type="button" className="btn" onClick={() => onAsk(fixed)}>Search {help!.did_you_mean} instead</button>}
          {area && help?.covered && <button type="button" className="btn" onClick={() => onAsk(`How much is rent in ${area}?`)}>See all of {area}</button>}
          {area && !help?.covered && !help?.did_you_mean && <button type="button" className="btn" onClick={() => onAsk(`What is it like to live in ${area}, Lagos?`)}>What is {area} like to live in?</button>}
          {others.map((a) => <button key={a} type="button" className="chip" onClick={() => onAsk(`How much is rent in ${a}?`)}>Try {a}</button>)}
        </div>
      )}
    </section>
  );
}

const IN_SCOPE_EXAMPLES = ["2 bedroom rent in Yaba", "Self contain under ₦1M", "Lekki or Ajah for a 3 bedroom?", "Is agent fee negotiable?"];

/** Wale's hello (or "you're welcome"): who he is, what he can do, and questions to tap. */
export function GreetingState({ greeting, onAsk }: { greeting: Greeting; onAsk?: (q: string) => void }) {
  return (
    <section className="card rise flex gap-4 p-6" aria-label="Greeting">
      <span aria-hidden className="hidden shrink-0 sm:block"><MascotHead size={48} /></span>
      <div className="min-w-0">
        <h2 className="text-[18px] font-bold">{greeting.title}</h2>
        {greeting.body && <p className="muted mt-1 text-[15px] leading-relaxed">{greeting.body}</p>}
        {greeting.can.length > 0 && (
          <ul className="mt-3 space-y-1.5 text-[15px]">
            {greeting.can.map((c) => (
              <li key={c} className="flex gap-2"><span aria-hidden style={{ color: "var(--action)" }}>•</span><span>{c}</span></li>
            ))}
          </ul>
        )}
        {greeting.close && <p className="mt-3 text-[15px] font-semibold">{greeting.close}</p>}
        {onAsk && (
          <div className="mt-3 flex flex-wrap gap-2">
            {greeting.examples.map((q) => <button key={q} type="button" className="chip" onClick={() => onAsk(q)}>{q}</button>)}
          </div>
        )}
      </div>
    </section>
  );
}

export function OutOfScopeState({ message, onAsk, pidgin = false }: { message?: string; onAsk?: (q: string) => void; pidgin?: boolean }) {
  return (
    <section className="card rise p-6" aria-label="Out of scope">
      <h2 className="text-[18px] font-bold">{pidgin ? "Dis one no be wetin I fit help with" : "That is outside what I can help with"}</h2>
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
