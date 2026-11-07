import type { ChatAnswer } from "../types/api";

const BASE = import.meta.env.VITE_API_BASE_URL || "";

export function newRequestKey(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export async function ask(question: string, sessionId: string | null, requestKey: string): Promise<ChatAnswer> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Request-Key": requestKey },
    body: JSON.stringify({ query: question, session_id: sessionId, request_key: requestKey }),
  });
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  const body = await res.json();
  return (body.answer || body) as ChatAnswer;
}

export async function askLegacy(question: string, sessionId: string | null): Promise<{ answer: ChatAnswer }> {
  const key = newRequestKey();
  const answer = await ask(question, sessionId, key);
  return { answer };
}
