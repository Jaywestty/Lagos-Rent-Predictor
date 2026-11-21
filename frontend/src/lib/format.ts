export function formatNaira(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "No data available";
  return "NGN " + Math.round(value).toLocaleString("en-NG");
}

export function formatNairaRange(min: number | null, max: number | null): string {
  if (min == null && max == null) return "No data available";
  if (min == null) return formatNaira(max);
  if (max == null) return formatNaira(min);
  return `${formatNaira(min)} to ${formatNaira(max)}`;
}

/** Short naira for headlines and chart labels: ₦850k, ₦1.5M, ₦12M. */
export function nairaShort(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `₦${trim(value / 1e9)}B`;
  if (abs >= 1e6) return `₦${trim(value / 1e6)}M`;
  if (abs >= 1e3) return `₦${Math.round(value / 1e3)}k`;
  return `₦${Math.round(value)}`;
}

function trim(n: number): string {
  return n >= 100 ? String(Math.round(n)) : n.toFixed(n >= 10 ? 0 : 1).replace(/\.0$/, "");
}

export function timeAgo(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "date unknown";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "date unknown";
  const days = Math.floor((now.getTime() - t) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks} week${weeks > 1 ? "s" : ""} ago`;
  const months = Math.floor(days / 30);
  return `${months} month${months > 1 ? "s" : ""} ago`;
}

export const BUCKET_LABELS: Record<string, string> = {
  self_contain: "Self contain",
  one_bedroom: "1 bedroom",
  two_bedroom: "2 bedroom",
  three_bedroom: "3 bedroom",
};

export const SOURCE_LABELS: Record<string, string> = { npc: "NigeriaPropertyCentre", propertypro: "PropertyPro" };
