/** Ticker validation — mirrors `backend/app/models/common.py` (RNF-09). */
export const TICKER_RE = /^[A-Z0-9]{2,5}$/

export function normalizeTicker(raw: string): string {
  return raw.trim().toUpperCase()
}

export function isValidTicker(raw: string): boolean {
  return TICKER_RE.test(normalizeTicker(raw))
}
