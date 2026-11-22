import type { ReactNode } from "react";

/**
 * A small, safe Markdown renderer for AI answers: paragraphs, headings, bullet and numbered lists,
 * GitHub-style tables, **bold**, *italic*, `code`, http(s) links, and [Source N] / 【Source N】 citations.
 * It builds React elements only; nothing is ever injected as HTML.
 */

type Source = { title: string; url: string };

const INLINE = /(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|_[^_\s][^_]*_|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^)\s]+\)|【\s*Source\s*\d+\s*】|\[\s*Source\s*\d+\s*\])/g;

function inline(text: string, sources: Source[], key: string): ReactNode[] {
  const clean = text.replace(/\\([*_`[\]|#\\])/g, "$1");
  return clean.split(INLINE).filter((p) => p !== "").map((part, i) => {
    const k = `${key}-${i}`;
    let m = part.match(/^(?:\*\*|__)(.+)(?:\*\*|__)$/);
    if (m) return <strong key={k}>{inline(m[1], sources, k)}</strong>;
    m = part.match(/^`(.+)`$/);
    if (m) return <code key={k} className="rounded px-1 text-[0.92em]" style={{ background: "var(--line)" }}>{m[1]}</code>;
    m = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
    // Only link to pages we actually cited; any other link (for example one planted in a web page) stays plain text.
    if (m && !sources.some((s) => s.url === m![2])) return <span key={k}>{m[1]}</span>;
    if (m) return <a key={k} href={m[2]} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2" style={{ color: "var(--action)" }}>{m[1]}</a>;
    m = part.match(/Source\s*(\d+)/);
    if (m && /^[【[]/.test(part)) {
      const src = sources[Number(m[1]) - 1];
      return src
        ? <a key={k} href={src.url} target="_blank" rel="noopener noreferrer" title={src.title} className="mx-0.5 inline-block rounded-full px-1.5 align-baseline text-[11px] font-bold hover:underline" style={{ background: "var(--soft)", color: "var(--action)" }}>{m[1]}</a>
        : null;
    }
    m = part.match(/^(?:\*|_)(.+)(?:\*|_)$/);
    if (m) return <em key={k}>{m[1]}</em>;
    return <span key={k}>{part}</span>;
  });
}

const isTableRow = (l: string) => /^\s*\|.*\|\s*$/.test(l);
const isDivider = (l: string) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
const cells = (l: string) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

export function Markdown({ text, sources = [] }: { text: string; sources?: Source[] }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const key = `b${i}`;
    if (!line.trim() || /^\s*(-{3,}|\*{3,})\s*$/.test(line)) { i++; continue; }

    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (heading) {
      blocks.push(<p key={key} className="font-display text-[16px] font-bold">{inline(heading[1], sources, key)}</p>);
      i++;
      continue;
    }

    if (isTableRow(line) && i + 1 < lines.length && isDivider(lines[i + 1])) {
      const head = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && isTableRow(lines[i])) { rows.push(cells(lines[i])); i++; }
      blocks.push(
        <div key={key} className="overflow-x-auto rounded-xl border" style={{ borderColor: "var(--border)" }}>
          <table className="w-full min-w-[420px] text-left text-[14px]">
            <thead><tr style={{ background: "var(--bg)" }}>{head.map((h, c) => <th key={c} className="px-3 py-2 text-[12px] font-bold">{inline(h, sources, `${key}h${c}`)}</th>)}</tr></thead>
            <tbody>{rows.map((r, ri) => <tr key={ri} className="border-t" style={{ borderColor: "var(--line)" }}>{r.map((c, ci) => <td key={ci} className="px-3 py-2 align-top">{inline(c, sources, `${key}r${ri}c${ci}`)}</td>)}</tr>)}</tbody>
          </table>
        </div>,
      );
      continue;
    }

    const bullet = /^\s*(?:[-*•]|\d+[.)])\s+/;
    if (bullet.test(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const items: string[] = [];
      const sameKind = (l: string) => bullet.test(l) && /^\s*\d+[.)]/.test(l) === ordered;
      while (i < lines.length && sameKind(lines[i])) {
        let item = lines[i].replace(bullet, "");
        i++;
        while (i < lines.length && lines[i].trim() && !bullet.test(lines[i]) && /^\s{2,}/.test(lines[i])) { item += " " + lines[i].trim(); i++; }
        items.push(item);
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(
        <List key={key} className={`space-y-1.5 pl-5 marker:text-[color:var(--action)] ${ordered ? "list-decimal" : "list-disc"}`}>
          {items.map((it, n) => <li key={n} className="pl-1">{inline(it, sources, `${key}-${n}`)}</li>)}
        </List>,
      );
      continue;
    }

    const para: string[] = [line.trim()];
    i++;
    while (i < lines.length && lines[i].trim() && !bullet.test(lines[i]) && !isTableRow(lines[i]) && !/^\s*#{1,6}\s/.test(lines[i])) { para.push(lines[i].trim()); i++; }
    blocks.push(<p key={key}>{inline(para.join(" "), sources, key)}</p>);
  }
  return <div className="space-y-3 text-[15px] leading-relaxed">{blocks}</div>;
}
