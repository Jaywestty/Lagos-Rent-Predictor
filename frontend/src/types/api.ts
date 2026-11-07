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
};
export type SseEvent =
  | { type: "status"; request_id: string; text: string }
  | { type: "delta"; request_id: string; text: string }
  | { type: "final"; request_id: string; result: { answer: ChatAnswer } }
  | { type: "error"; request_id: string; message: string };
