export function stripPosition(min: number, median: number, max: number, scaleMin: number, scaleMax: number): { left: number; width: number; marker: number } {
  const range = Math.max(scaleMax - scaleMin, 1);
  const left = ((min - scaleMin) / range) * 100;
  const width = (((max - min) / range) * 100) || 0;
  const marker = ((median - scaleMin) / range) * 100;
  return { left: Math.max(0, Math.min(100, left)), width: Math.max(0, Math.min(100 - left, width)), marker: Math.max(0, Math.min(100, marker)) };
}

export function confidenceFor(count: number): "normal" | "limited" | "insufficient" {
  if (count >= 10) return "normal";
  if (count >= 4) return "limited";
  return "insufficient";
}

export function statusFor(count: number): string {
  if (count === 0) return "no_data";
  if (count <= 3) return "low_data";
  return "success";
}
