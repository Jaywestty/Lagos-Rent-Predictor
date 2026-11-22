import type { Advice } from "../../services/api";
import { Markdown } from "./Markdown";

export function AdviceAnswer({ advice }: { advice: Advice }) {
  return (
    <section className="card rise overflow-hidden" aria-label="Advice">
      <div className="space-y-4 p-5 md:p-7">
        <p className="kicker">General advice · from web sources</p>
        <Markdown text={advice.text} sources={advice.sources} />
      </div>
      <div className="space-y-3 border-t px-5 py-4 md:px-7" style={{ borderColor: "var(--line)", background: "var(--bg)" }}>
        <p className="text-[13px] leading-relaxed">
          This is general guidance from the web, not our listings. For prices, ask about an area and size, for example <strong>“2 bedroom rent in Yaba”</strong>.
        </p>
        {advice.sources.length > 0 && (
          <details className="text-[13px]">
            <summary className="cursor-pointer font-semibold" style={{ color: "var(--action)" }}>Sources ({advice.sources.length})</summary>
            <ol className="mt-2 list-decimal space-y-1 pl-5">
              {advice.sources.map((s) => <li key={s.url}><a href={s.url} target="_blank" rel="noopener noreferrer" className="hover:underline">{s.title || s.url}</a></li>)}
            </ol>
          </details>
        )}
      </div>
    </section>
  );
}
