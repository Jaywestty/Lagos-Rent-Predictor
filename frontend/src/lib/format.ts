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
