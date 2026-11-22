import { createContext, useContext } from "react";
import { useQuery } from "@tanstack/react-query";
import { getAreas } from "../services/api";
import type { AreaSummary } from "../types/api";

export type SiteStats = { listings: number; areas: number; newest: string | null; top: AreaSummary[] };

/** Live numbers for the header, footer and welcome screen, from the same /api/areas call the app already makes. */
export function useSiteStats() {
  return useQuery({
    queryKey: ["areas"],
    queryFn: getAreas,
    staleTime: 5 * 60_000,
    retry: false,
    select: (items): SiteStats => ({
      listings: items.reduce((n, a) => n + a.listing_count, 0),
      areas: items.length,
      newest: items.map((a) => a.latest_listing_at).filter((d): d is string => Boolean(d)).sort().at(-1) ?? null,
      top: items.filter((a) => a.median_ngn != null).slice(0, 14),
    }),
  });
}

/** Lets a page tell the layout it is showing a full-height chat (the footer steps aside). */
export const ChromeContext = createContext<{ setChatActive: (active: boolean) => void }>({ setChatActive: () => undefined });
export const useChrome = () => useContext(ChromeContext);
