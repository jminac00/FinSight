/**
 * Cookie / client-storage consent model (ADR-0002, art. 22 LSSI-CE, AEPD).
 *
 * Three categories:
 *  - "necessary": always active, no consent required (the consent decision
 *    itself is stored under this category — RNF-35).
 *  - "functional": last ticker, disclaimer acknowledgement, theme (RNF-37).
 *  - "analytics": usage measurement (RNF-38).
 *
 * Per RNF-39, the same rules apply to any client storage (cookies, localStorage,
 * sessionStorage, IndexedDB), not just cookies.
 */
export type OptionalCategory = 'functional' | 'analytics'

export type ConsentState = {
  /** Whether the user has made an explicit choice (banner dismissed). */
  decided: boolean
  functional: boolean
  analytics: boolean
  /** ISO timestamp of the decision, for audit/expiry. */
  timestamp: string | null
}

export const DEFAULT_CONSENT: ConsentState = {
  decided: false,
  functional: false,
  analytics: false,
  timestamp: null,
}
