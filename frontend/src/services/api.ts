import type { AreaGuide, AreaStats, AreaSummary, ChatAnswer, CompareResult, Favorite, Listing, SavedSearch, Trend } from "../types/api";

const BASE = import.meta.env.VITE_API_BASE_URL || "";

/** Anonymous per-device id used for saved searches and favourites. */
export function clientId(): string {
  try {
    let id = localStorage.getItem("lagos-client-id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("lagos-client-id", id);
    }
    return id;
  } catch {
    return "00000000-0000-4000-8000-000000000000";
  }
}

export function newRequestKey(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", "X-Client-Id": clientId(), ...(init.headers || {}) },
  });
  if (!res.ok) {
    let detail = `Request failed: ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep default */
    }
    throw Object.assign(new Error(detail), { status: res.status });
  }
  return res.json() as Promise<T>;
}

type ChatBody = { query_type?: string; sources?: { title: string; url: string }[]; answer?: ChatAnswer; results?: Listing[]; areas?: { results: Listing[] }[]; options?: { results: Listing[] }[] } & Partial<ChatAnswer>;

export type Advice = { text: string; sources: { title: string; url: string }[] };
export type AskResult = { answer: ChatAnswer; listings: Listing[]; advice: Advice | null };

export async function ask(question: string, sessionId: string | null, requestKey: string): Promise<AskResult> {
  const body = await call<ChatBody>("/api/chat", {
    method: "POST",
    headers: { "X-Request-Key": requestKey },
    body: JSON.stringify({ query: question, session_id: sessionId, request_key: requestKey }),
  });
  const listings = [
    ...(body.results || []),
    ...(body.areas || []).flatMap((a) => a.results || []),
    ...(body.options || []).flatMap((o) => o.results || []),
  ];
  const seen = new Set<string>();
  return {
    answer: (body.answer || body) as ChatAnswer,
    advice: body.query_type === "advice" ? { text: body.answer?.message || "", sources: body.sources || [] } : null,
    listings: listings.filter((l) => l?.url && !seen.has(l.url) && seen.add(l.url)),
  };
}

export const getAreas = () => call<{ items: AreaSummary[] }>("/api/areas").then((r) => r.items);
export const getArea = (area: string) => call<AreaStats & { trend: Trend }>(`/api/areas/${encodeURIComponent(area)}`);
export const getTrend = (area: string, bucket: string) => call<Trend>(`/api/areas/${encodeURIComponent(area)}/trends?bucket=${bucket}`);
export const getAreaListings = (area: string, beds?: number | null, limit = 9) =>
  call<{ items: Listing[] }>(`/api/areas/${encodeURIComponent(area)}/listings?limit=${limit}${beds != null ? `&beds=${beds}` : ""}`).then((r) => r.items);
export const getGuide = (area: string) => call<AreaGuide>(`/api/areas/${encodeURIComponent(area)}/guide`);
export const getCompare = (areas: string[], beds: number | null) =>
  call<CompareResult>(`/api/compare?areas=${encodeURIComponent(areas.join(","))}${beds ? `&beds=${beds}` : ""}`);

export const getSaved = () => call<{ items: SavedSearch[] }>("/api/me/saved-searches").then((r) => r.items);
export const saveSearch = (s: { label: string; query?: string | null; area?: string | null; bedrooms?: number | null; max_budget_ngn?: number | null }) =>
  call<{ id: number }>("/api/me/saved-searches", { method: "POST", body: JSON.stringify(s) });
export const deleteSaved = (id: number) => call<{ ok: boolean }>(`/api/me/saved-searches/${id}`, { method: "DELETE" });

export const getFavorites = () => call<{ items: Favorite[] }>("/api/me/favorites").then((r) => r.items);
export const addFavorite = (l: Listing) =>
  call<{ id: number }>("/api/me/favorites", {
    method: "POST",
    body: JSON.stringify({ url: l.url, title: l.title, price_text: l.price, annual_price_ngn: l.annual_price_ngn ?? null, area: l.area, bedrooms: l.bedrooms, source: l.source ?? null }),
  });
export const deleteFavorite = (id: number) => call<{ ok: boolean }>(`/api/me/favorites/${id}`, { method: "DELETE" });

/** Fire-and-forget usage event. Never throws. */
export function track(feature: string, detail: Record<string, string | number | boolean | null> = {}) {
  call("/api/events", { method: "POST", body: JSON.stringify({ feature, detail }) }).catch(() => undefined);
}
