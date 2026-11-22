export type RangeStat = { min: number | null; max: number | null; median: number | null; count: number };
export type Breakdown = { self_contain: RangeStat | null; one_bedroom: RangeStat | null; two_bedroom: RangeStat | null; three_bedroom: RangeStat | null };
export type ChatAnswer = {
  version: number;
  status: "success" | "low_data" | "no_data" | "clarification" | "out_of_scope" | "error";
  location: string | null;
  area: string | null;
  property_type: string | null;
  currency: "NGN";
  rent_period: "yearly";
  headline: { min: number | null; max: number | null } | null;
  breakdown: Breakdown | null;
  median: number | null;
  listing_count: number;
  range_method: "percentile" | "min_max";
  data_as_of: string | null;
  last_updated_days: number | null;
  confidence: "normal" | "limited" | "insufficient";
  clarification: { question: string; options: string[] } | null;
  follow_ups: string[];
  request_id: string;
  latency_ms: number;
  message: string;
  comparison?: { options: ComparisonOptionStat[] } | null;
  budget?: BudgetInfo | null;
  no_data?: NoDataHelp | null;
};
export type NoDataHelp = { area: string | null; covered?: boolean; area_listings?: number; did_you_mean?: string | null; suggestions?: string[] };
export type ComparisonOptionStat = { label: string; area: string | null; beds: number | null; property_type: string | null; median: number | null; min: number | null; max: number | null; count: number; confidence: "normal" | "limited" | "insufficient"; caveat: string | null };
export type BudgetInfo = { amount: number; period: "year" | "month"; yearly: number; within_count: number; areas: { area: string; count: number }[]; stretch_count: number };
export type SseEvent =
  | { type: "status"; request_id: string; text: string }
  | { type: "delta"; request_id: string; text: string }
  | { type: "final"; request_id: string; result: { answer: ChatAnswer } }
  | { type: "error"; request_id: string; message: string };

export type Listing = {
  title: string | null;
  price: string;
  area: string | null;
  subarea: string | null;
  bedrooms: number | null;
  url: string;
  id?: number | null;
  source?: string | null;
  bathrooms?: number | null;
  property_type?: string | null;
  annual_price_ngn?: number | null;
  listed_at?: string | null;
};
export type Summary = { min: number | null; max: number | null; median: number | null; count: number; range_method: "percentile" | "min_max" };
export type BucketKey = "self_contain" | "one_bedroom" | "two_bedroom" | "three_bedroom";
export type AreaStats = {
  area: string;
  listing_count: number;
  status: string;
  confidence: "normal" | "limited" | "insufficient";
  overall: Summary | null;
  breakdown: Record<BucketKey, Summary | null>;
  data_as_of: string | null;
};
export type TrendPoint = { week_start: string; listing_count: number; median_ngn: number | null; p10_ngn: number | null; p90_ngn: number | null };
export type Trend = { area: string; bucket: string; label: string; points: TrendPoint[]; change_pct: number | null; weeks_with_data: number };
export type AreaSummary = { area: string; listing_count: number; median_ngn: number | null; latest_listing_at: string | null };
export type GuideSection = { key: string; title: string; text: string | null; sources: number[] };
export type AreaGuide = {
  area: string;
  stats: AreaStats;
  guide: { sections: GuideSection[]; watch_outs: string[]; watch_out_sources: number[] } | null;
  sources?: { n: number; title: string; url: string }[];
  generated_at?: string;
  stale?: boolean;
  guide_error?: string;
};
export type CompareResult = { bucket: string; label: string; areas: (AreaStats & { focus: Summary | null; trend: Trend })[] };
export type SavedSearch = { id: number; label: string; query: string | null; area: string | null; bedrooms: number | null; max_budget_ngn: number | null; created_at: string; current?: { listing_count: number; median_ngn: number | null; data_as_of: string | null } };
export type Favorite = { id: number; url: string; title: string | null; price_text: string | null; annual_price_ngn: number | null; area: string | null; bedrooms: number | null; source: string | null; created_at: string; still_listed: boolean; listed_at: string | null };
export type ChatThreadSummary = { id: string; title: string; message_count: number; created_at: string; updated_at: string; last_question: string | null };
export type ChatThreadMessage = { id: number; question: string; created_at: string; payload: { answer: ChatAnswer; listings: Listing[]; advice?: { text: string; sources: { title: string; url: string }[] } } };
export type ChatThread = { id: string; title: string; message_count: number; created_at: string; updated_at: string; messages: ChatThreadMessage[] };
