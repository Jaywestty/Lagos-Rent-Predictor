import type { Advice } from "../../services/api";

/** Renders **bold** spans and turns 【Source 2】 style markers into [2] links. No HTML is injected. */
function Inline({ text, sources }: { text: string; sources: Advice["sources"] }) {
  const parts = text.split(/(\*\*[^*]+\*\*|【Source \d+】|\[Source \d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const bold = part.match(/^\*\*([^*]+)\*\*$/);
        if (bold) return <strong key={i}>{bold[1]}</strong>;
        const ref = part.match(/(?:【|\[)Source (\d+)(?:】|\])/);
        if (ref) {
          const src = sources[Number(ref[1]) - 1];
          return src ? <a key={i} href={src.url} target="_blank" rel="noopener noreferrer" className="badge mx-0.5 align-baseline hover:underline" title={src.title}>[{ref[1]}]</a> : null;
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

export function AdviceAnswer({ advice }: { advice: Advice }) {
  const paragraphs = advice.text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  return (
    <section className="card rise p-4 md:p-6" aria-label="Advice">
      <p className="muted text-[12px] font-semibold uppercase tracking-wide">General advice · from web sources</p>
      <div className="mt-2 space-y-3 text-[15px] leading-relaxed">
        {paragraphs.map((p, i) => <p key={i}><Inline text={p.replace(/^[-*]\s+/gm, "• ")} sources={advice.sources} /></p>)}
      </div>
      <p className="mt-3 rounded-lg p-3 text-[13px]" style={{ background: "var(--soft)" }}>
        This is general guidance from the web, not our listings data. For rent prices, ask about an area, for example “2 bedroom rent in Yaba”.
      </p>
      {advice.sources.length > 0 && (
        <details className="mt-3 text-[13px]">
          <summary className="cursor-pointer font-semibold">Sources ({advice.sources.length})</summary>
          <ol className="mt-2 list-decimal space-y-1 pl-5">{advice.sources.map((s) => <li key={s.url}><a href={s.url} target="_blank" rel="noopener noreferrer" className="hover:underline">{s.title || s.url}</a></li>)}</ol>
        </details>
      )}
    </section>
  );
}
