/**
 * Ticker validation — mirrors `backend/app/models/common.py` (RNF-09).
 *
 * International symbols carry exchange suffixes and class separators
 * (e.g. REP.MC, ASML.AS, 7203.T, BRK-B), so the US-only 2-5 alphanumeric
 * rule is relaxed for universal (MSCI World) coverage.
 */
export const TICKER_RE = /^[A-Z0-9][A-Z0-9.-]{0,14}$/

export function normalizeTicker(raw: string): string {
  return raw.trim().toUpperCase()
}

export function isValidTicker(raw: string): boolean {
  return TICKER_RE.test(normalizeTicker(raw))
}
