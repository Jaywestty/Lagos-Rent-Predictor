import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteAllThreads, deleteThread, getThreads } from "../../services/api";
import { timeAgo } from "../../lib/format";
import type { ChatThreadSummary } from "../../types/api";

export function useThreads(enabled = true) {
  return useQuery({ queryKey: ["threads"], queryFn: getThreads, enabled, staleTime: 10_000, retry: false });
}

function groupOf(t: ChatThreadSummary): string {
  const days = (Date.now() - new Date(t.updated_at).getTime()) / 86_400_000;
  if (days < 1) return "Today";
  if (days < 7) return "This week";
  return "Earlier";
}

type ListProps = { activeId: string | null; onOpen: (id: string) => void; onNew: () => void; enabled?: boolean };

/** This device's past chats, grouped by date, with delete and clear-all. Used in the sidebar and the drawer. */
export function ChatList({ activeId, onOpen, onNew, enabled = true }: ListProps) {
  const qc = useQueryClient();
  const [confirmClear, setConfirmClear] = useState(false);
  const threads = useThreads(enabled);
  const remove = useMutation({ mutationFn: deleteThread, onSuccess: () => qc.invalidateQueries({ queryKey: ["threads"] }) });
  const clear = useMutation({
    mutationFn: deleteAllThreads,
    onSuccess: () => { setConfirmClear(false); qc.invalidateQueries({ queryKey: ["threads"] }); onNew(); },
  });
  const items = threads.data ?? [];
  const groups = ["Today", "This week", "Earlier"].map((g) => ({ g, items: items.filter((t) => groupOf(t) === g) })).filter((x) => x.items.length);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        {threads.isPending && <div className="space-y-2 p-1">{[0, 1, 2].map((i) => <div key={i} className="skeleton h-14" />)}</div>}
        {threads.isError && <p className="p-3 text-[14px]" role="alert">Could not load your chats.</p>}
        {threads.data && !items.length && <p className="muted p-3 text-[14px] leading-relaxed">No chats yet. Your questions and answers are kept here, on this device only.</p>}
        {groups.map(({ g, items: list }) => (
          <div key={g} className="mb-3">
            <p className="kicker px-3 pb-1 pt-2 text-[11px]">{g}</p>
            <ul className="space-y-0.5">
              {list.map((t) => (
                <li key={t.id} className="group flex items-stretch rounded-lg" style={t.id === activeId ? { background: "var(--soft)" } : undefined}>
                  <button type="button" className="min-w-0 flex-1 rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-[color:var(--soft)]" onClick={() => onOpen(t.id)} aria-current={t.id === activeId ? "true" : undefined}>
                    <p className="truncate text-[14px] font-semibold">{t.title}</p>
                    <p className="muted truncate text-[12px]">{t.message_count} question{t.message_count === 1 ? "" : "s"} · {timeAgo(t.updated_at)}</p>
                  </button>
                  <button type="button" aria-label={`Delete chat: ${t.title}`} className="muted grid w-10 place-items-center rounded-lg opacity-50 transition-opacity hover:opacity-100 focus:opacity-100 group-hover:opacity-100" disabled={remove.isPending}
                    onClick={() => { remove.mutate(t.id); if (t.id === activeId) onNew(); }}>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></svg>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      {items.length > 0 && (
        <div className="border-t px-3 pt-3 text-[13px]" style={{ borderColor: "var(--border)" }}>
          {!confirmClear ? (
            <button type="button" className="muted hover:underline" onClick={() => setConfirmClear(true)}>Clear all chats</button>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <span>Delete all {items.length} chats? This cannot be undone.</span>
              <button type="button" className="font-semibold" style={{ color: "var(--danger)" }} disabled={clear.isPending} onClick={() => clear.mutate()}>Delete all</button>
              <button type="button" className="muted" onClick={() => setConfirmClear(false)}>Cancel</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

type DrawerProps = { open: boolean; onClose: () => void; activeId: string | null; onOpen: (id: string) => void; onNew: () => void };

/** Slide-over version of the chat list for phones. */
export function ChatHistory({ open, onClose, activeId, onOpen, onNew }: DrawerProps) {
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    panel.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-30" role="dialog" aria-modal="true" aria-label="Chat history">
      <button type="button" aria-label="Close chat history" className="fade-in absolute inset-0 cursor-default" style={{ background: "rgba(15, 21, 19, .4)" }} onClick={onClose} />
      <div ref={panel} tabIndex={-1} className="slide-in absolute inset-y-0 left-0 flex w-[min(340px,88vw)] flex-col gap-3 border-r p-3 pb-4 outline-none" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="flex items-center justify-between gap-2 px-1 pt-1">
          <h2 className="text-[18px] font-bold">Your chats</h2>
          <button type="button" className="btn-ghost grid w-11 place-items-center px-0" aria-label="Close" onClick={onClose}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>
        <button type="button" className="btn" onClick={() => { onNew(); onClose(); }}>+ New chat</button>
        <ChatList activeId={activeId} onOpen={(id) => { onOpen(id); onClose(); }} onNew={onNew} enabled={open} />
      </div>
    </div>
  );
}
