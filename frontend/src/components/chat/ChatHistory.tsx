import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteAllThreads, deleteThread, getThreads } from "../../services/api";
import { timeAgo } from "../../lib/format";

type Props = { open: boolean; onClose: () => void; activeId: string | null; onOpen: (id: string) => void; onNew: () => void };

/** Slide-over list of this device's past chats: open one, delete one, or clear all. */
export function ChatHistory({ open, onClose, activeId, onOpen, onNew }: Props) {
  const qc = useQueryClient();
  const [confirmClear, setConfirmClear] = useState(false);
  const panel = useRef<HTMLDivElement>(null);
  const threads = useQuery({ queryKey: ["threads"], queryFn: getThreads, enabled: open, staleTime: 10_000 });
  const remove = useMutation({ mutationFn: deleteThread, onSuccess: () => qc.invalidateQueries({ queryKey: ["threads"] }) });
  const clear = useMutation({
    mutationFn: deleteAllThreads,
    onSuccess: () => { setConfirmClear(false); qc.invalidateQueries({ queryKey: ["threads"] }); onNew(); },
  });

  useEffect(() => {
    if (!open) return;
    panel.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  const items = threads.data ?? [];

  return (
    <div className="fixed inset-0 z-30" role="dialog" aria-modal="true" aria-label="Chat history">
      <button type="button" aria-label="Close chat history" className="fade-in absolute inset-0 cursor-default" style={{ background: "rgba(16, 23, 19, .35)" }} onClick={onClose} />
      <div ref={panel} tabIndex={-1} className="slide-in absolute inset-y-0 left-0 flex w-[min(360px,88vw)] flex-col border-r outline-none" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="flex items-center justify-between gap-2 border-b p-4" style={{ borderColor: "var(--border)" }}>
          <h2 className="text-[17px] font-semibold">Your chats</h2>
          <div className="flex gap-2">
            <button type="button" className="btn" onClick={() => { onNew(); onClose(); }}>New chat</button>
            <button type="button" className="btn-ghost" aria-label="Close" onClick={onClose}>✕</button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {threads.isPending && <div className="space-y-2 p-2">{[0, 1, 2].map((i) => <div key={i} className="skeleton h-14" />)}</div>}
          {threads.isError && <p className="p-3 text-[14px]" role="alert">Could not load your chats.</p>}
          {threads.data && !items.length && <p className="muted p-3 text-[14px]">No chats yet. Your questions and answers are kept here on this device.</p>}
          <ul className="stagger space-y-1">
            {items.map((t) => (
              <li key={t.id} className="group flex items-stretch gap-1 rounded-lg" style={t.id === activeId ? { background: "var(--soft)" } : undefined}>
                <button type="button" className="min-w-0 flex-1 rounded-lg p-3 text-left transition-colors hover:bg-[color:var(--soft)]" onClick={() => { onOpen(t.id); onClose(); }} aria-current={t.id === activeId ? "true" : undefined}>
                  <p className="truncate text-[14px] font-semibold">{t.title}</p>
                  <p className="muted truncate text-[12px]">{t.message_count} question{t.message_count === 1 ? "" : "s"} · {timeAgo(t.updated_at)}{t.last_question && t.message_count > 1 ? ` · ${t.last_question}` : ""}</p>
                </button>
                <button type="button" aria-label={`Delete chat: ${t.title}`} className="muted rounded-lg px-3 text-[13px] opacity-60 hover:opacity-100 focus:opacity-100" disabled={remove.isPending}
                  onClick={() => { remove.mutate(t.id); if (t.id === activeId) onNew(); }}>Delete</button>
              </li>
            ))}
          </ul>
        </div>
        {items.length > 0 && (
          <div className="border-t p-3 text-[13px]" style={{ borderColor: "var(--border)" }}>
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
    </div>
  );
}
