import { beforeEach, describe, expect, it } from 'vitest'
import { getStored, readConsent, removeStored, setStored, writeConsent } from './storage'
import type { ConsentState } from './types'

function consent(overrides: Partial<ConsentState>): ConsentState {
  return {
    decided: true,
    functional: false,
    analytics: false,
    timestamp: new Date().toISOString(),
    ...overrides,
  }
}

beforeEach(() => {
  // Clear cookie and storage between tests.
  document.cookie = 'finsight_consent=; Max-Age=0; Path=/'
  window.localStorage.clear()
})

describe('consent-gated storage (RNF-39)', () => {
  it('refuses to write when the category is not consented', () => {
    writeConsent(consent({ functional: false }))
    expect(setStored('finsight.theme', 'dark', 'functional')).toBe(false)
    expect(window.localStorage.getItem('finsight.theme')).toBeNull()
  })

  it('writes and reads when the category is consented', () => {
    writeConsent(consent({ functional: true }))
    expect(setStored('finsight.theme', 'dark', 'functional')).toBe(true)
    expect(getStored('finsight.theme', 'functional')).toBe('dark')
  })

  it('does not expose analytics storage without analytics consent', () => {
    writeConsent(consent({ functional: true, analytics: false }))
    expect(setStored('finsight.metric', '1', 'analytics')).toBe(false)
  })

  it('purges stored data when consent is withdrawn', () => {
    writeConsent(consent({ functional: true }))
    setStored('finsight.theme', 'dark', 'functional')
    writeConsent(consent({ functional: false }))
    expect(window.localStorage.getItem('finsight.theme')).toBeNull()
  })
})

describe('readConsent', () => {
  it('defaults to no consent before any decision', () => {
    const state = readConsent()
    expect(state.decided).toBe(false)
    expect(state.functional).toBe(false)
    expect(state.analytics).toBe(false)
  })

  it('round-trips the persisted decision', () => {
    writeConsent(consent({ functional: true, analytics: true }))
    const state = readConsent()
    expect(state).toMatchObject({ decided: true, functional: true, analytics: true })
  })
})

describe('removeStored', () => {
  it('removes a key regardless of consent', () => {
    writeConsent(consent({ functional: true }))
    setStored('finsight.theme', 'dark', 'functional')
    removeStored('finsight.theme')
    expect(window.localStorage.getItem('finsight.theme')).toBeNull()
  })
})
