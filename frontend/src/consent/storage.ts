import { DEFAULT_CONSENT, type ConsentState, type OptionalCategory } from './types'

const CONSENT_COOKIE = 'finsight_consent'
const CONSENT_MAX_AGE = 60 * 60 * 24 * 180 // 180 days, in seconds

/** localStorage keys owned by each optional category, purged when consent drops. */
const CATEGORY_KEYS: Record<OptionalCategory, string[]> = {
  functional: ['finsight.lastTicker', 'finsight.theme', 'finsight.disclaimerAck'],
  analytics: [],
}

function readCookie(name: string): string | null {
  if (typeof document === 'undefined') return null
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`))
  return match ? decodeURIComponent(match[1]) : null
}

/**
 * The consent decision is a strictly-necessary technical cookie (RNF-35): it does
 * not itself require prior consent and keeps the banner from reappearing.
 */
export function readConsent(): ConsentState {
  const raw = readCookie(CONSENT_COOKIE)
  if (!raw) return DEFAULT_CONSENT
  try {
    const parsed = JSON.parse(raw) as Partial<ConsentState>
    return {
      decided: Boolean(parsed.decided),
      functional: Boolean(parsed.functional),
      analytics: Boolean(parsed.analytics),
      timestamp: parsed.timestamp ?? null,
    }
  } catch {
    return DEFAULT_CONSENT
  }
}

export function writeConsent(state: ConsentState): void {
  if (typeof document === 'undefined') return
  const value = encodeURIComponent(JSON.stringify(state))
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  document.cookie = `${CONSENT_COOKIE}=${value}; Max-Age=${CONSENT_MAX_AGE}; Path=/; SameSite=Lax${secure}`
  // Enforce RNF-39: drop any stored data whose category is no longer consented.
  purgeWithdrawn(state)
}

function purgeWithdrawn(state: ConsentState): void {
  ;(Object.keys(CATEGORY_KEYS) as OptionalCategory[]).forEach((category) => {
    if (!state[category]) {
      CATEGORY_KEYS[category].forEach((key) => safeRemove(key))
    }
  })
}

function safeRemove(key: string): void {
  try {
    window.localStorage.removeItem(key)
  } catch {
    /* storage unavailable (private mode / disabled) — nothing to clean up */
  }
}

/**
 * Gated localStorage write. No-ops (and refuses to store) when the owning
 * category has not been consented (RNF-39). Returns whether the write happened.
 */
export function setStored(key: string, value: string, category: OptionalCategory): boolean {
  if (!readConsent()[category]) return false
  try {
    window.localStorage.setItem(key, value)
    return true
  } catch {
    return false
  }
}

/** Reads stored data only if the category is currently consented. */
export function getStored(key: string, category: OptionalCategory): string | null {
  if (!readConsent()[category]) return null
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

export function removeStored(key: string): void {
  safeRemove(key)
}
