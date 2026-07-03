/**
 * Curated technical indicators shown to the user, grouped conceptually by
 * block (momentum, trend, risk/stability, confirmation). The backend's
 * indicator bag also contains internal computation artifacts (z-scores,
 * normalization internals, raw/duplicate scores, signal/summary already
 * shown elsewhere in the report) that are intentionally excluded here: they
 * have no meaningful explanation for the end user.
 */
export const TECHNICAL_INDICATOR_KEYS = [
  'momentum_12_1',
  'momentum_6_1',
  'vol_12m',
  'price',
  'ma_200',
  'distance_to_ma200',
  'max_drawdown_126d',
  'downside_volatility_126d',
  'current_volume',
  'average_volume_20',
  'relative_volume_20',
  'donchian_position',
] as const

/** Keep only the curated indicator keys present in the (flattened) input, in display order. */
export function pickTechnicalIndicators(flat: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const key of TECHNICAL_INDICATOR_KEYS) {
    if (key in flat) out[key] = flat[key]
  }
  return out
}
