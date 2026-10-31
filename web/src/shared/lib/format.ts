export const pct = (v: number | null | undefined, digits = 0) => (v == null ? 'n/a' : `${(100 * v).toFixed(digits)}%`);
export const num = (v: number | null | undefined, digits = 1) => (v == null ? 'n/a' : v.toFixed(digits));
export const signed = (v: number, digits = 1, unit = '') =>
  `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(digits)}${unit}`;
